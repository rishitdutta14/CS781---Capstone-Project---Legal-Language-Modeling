"""Explicit deadline revision: BF16 compute, complete cases and required methods.

The existing FP32 sources and checkpoint bindings are not modified. Small
cases use one joint graph; larger cases use the tested exact replay path.
Neither policy samples, caps or drops document chunks.
"""

import contextlib
from pathlib import Path
import time

import numpy as np
import torch

from src.phase1_full_corpus_prepare import atomic_json,file_sha256
from src.phase2_adaptation import LoRMAPlusLinear,frozen_parameter_sha256
from src.phase3_model import load_backbone
from src.phase3_rebuilt_model import load_rebuilt_backbone
from src.phase4_data import CaseData,strict_json
from src.phase4_model import (CaseClassifier,backward_case as replay_backward,
    batches,case_loss,predict_case,trainability)
import src.phase4_training as training


class DeadlineClassifier(CaseClassifier):
    def __init__(self,backbone,architecture,method,precision='bf16',partial_layers=2):
        super().__init__(backbone,'full_fine_tuning',head='linear',architecture=architecture)
        self.precision=precision;self.method=method
        if method=='partial_layerwise':
            backbone.requires_grad_(False)
            if architecture=='encoder_decoder':
                for stack in (backbone.encoder,backbone.decoder):
                    if not 0<partial_layers<len(stack):raise ValueError('Invalid partial stack depth')
                    for block in stack[-partial_layers:]:block.requires_grad_(True)
                backbone.encoder_norm.requires_grad_(True);backbone.decoder_norm.requires_grad_(True)
            else:
                if not 0<partial_layers<len(backbone.blocks):raise ValueError('Invalid partial decoder depth')
                for block in backbone.blocks[-partial_layers:]:block.requires_grad_(True)
                backbone.ln_f.requires_grad_(True)
        elif method=='lorma_plus':
            backbone.requires_grad_(False)
            for index,block in enumerate(backbone.blocks):
                targets=('q_proj','v_up_proj') if hasattr(block.attn,'kv_down_proj') else ('q_proj','v_proj')
                for name in targets:
                    setattr(block.attn,name,LoRMAPlusLinear(getattr(block.attn,name),8,8.))
                    self.targets.append(f'backbone.blocks.{index}.attn.{name}')
        elif method not in ('full_fine_tuning','knowledge_distillation'):
            raise ValueError('Unannounced adaptation method')

    def chunk_vectors(self,ids,lengths):
        enabled=self.precision=='bf16' and ids.device.type=='cuda'
        with torch.autocast(ids.device.type,dtype=torch.bfloat16,enabled=enabled):
            return super().chunk_vectors(ids,lengths)


def build(config,device):
    if config['architecture'] in ('dense','encoder_decoder'):
        backbone=load_rebuilt_backbone(config['checkpoint'],config['model_config'],config['checkpoint_sha256'])
    else:backbone=load_backbone(config['checkpoint'],config['model_config'])
    return DeadlineClassifier(backbone,config['architecture'],config['method'],config['compute_precision'],
        config['partial_layers']).to(device)


def backward_case(model,chunks,label,denominator,microbatch=4,teacher=None,
                  temperature=2.,kd_weight=.5,class_weights=None):
    if len(chunks)>microbatch:
        return replay_backward(model,chunks,label,denominator,microbatch,teacher,
            temperature,kd_weight,class_weights)
    if not model.training or denominator<1:raise ValueError('Training and actual positive denominator required')
    device=next(model.parameters()).device
    ids,lengths=next(batches(chunks,microbatch,device))
    vectors=model.chunk_vectors(ids,lengths)
    # Hidden vectors, pooling, head, CE and KL remain FP32 outside autocast.
    logits=model.classifier(vectors.mean(0))
    loss,ce,kl=case_loss(logits,label,teacher,temperature,kd_weight,class_weights)
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite case loss')
    (loss/denominator).backward()
    return {'loss':float(loss.detach()),'ce':float(ce),'kl':float(kl),'chunks':len(chunks)}


@contextlib.contextmanager
def training_functions():
    previous=(training.build,training.backward_case)
    training.build=build;training.backward_case=backward_case
    try:yield
    finally:training.build,training.backward_case=previous


def gradient_metrics(reference,actual):
    norm_a=norm_b=dot=error=0.
    projections=[]
    for (name,a),(_,b) in zip(reference,actual):
        a=a.double().flatten();b=b.double().flatten()
        aa=float(a.dot(a));bb=float(b.dot(b));ab=float(a.dot(b))
        norm_a+=aa;norm_b+=bb;dot+=ab;error+=float((a-b).square().sum())
        if any(part in name for part in ('q_proj','k_proj','qkv','k_up_proj')) and aa>1e-16 and bb>1e-16:
            projections.append(ab/(aa*bb)**.5)
    return {'gradient_cosine':dot/max((norm_a*norm_b)**.5,1e-30),
        'relative_gradient_L2_error':(error/max(norm_a,1e-30))**.5,
        'FP32_gradient_norm':norm_a**.5,'BF16_gradient_norm':norm_b**.5,
        'minimum_query_key_projection_cosine':min(projections) if projections else None}


def precision_gate(model,data):
    """Compare full-context gradients, not merely finite losses or short prefixes."""
    device=next(model.parameters()).device
    checks=[];model.eval()
    # Full first/middle chunks plus a short chunk from a separate case. All TRAIN.
    first=data.chunks('train',0)
    probes=[(first[0],data.rows['train'][0]['label']),
        (first[len(first)//2],data.rows['train'][0]['label'])]
    short_index=next(i for i,row in enumerate(data.rows['train']) if row['tokens']<512)
    probes.append((data.chunks('train',short_index)[0],data.rows['train'][short_index]['label']))
    for chunk,label in probes:
        ids,lengths=next(batches([chunk],1,device));gradients=[];losses=[]
        for precision in ('fp32','bf16'):
            model.precision=precision
            logits=model.classifier(model.chunk_vectors(ids,lengths)[0])
            loss=case_loss(logits,label)[0]
            named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
            values=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
            gradients.append([(name,g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu'))
                for (name,p),g in zip(named,values)])
            losses.append(float(loss.detach()))
        metrics=gradient_metrics(*gradients)
        metrics.update(tokens=len(chunk),FP32_loss=losses[0],BF16_loss=losses[1])
        qk=metrics['minimum_query_key_projection_cosine']
        metrics['passed']=(all(np.isfinite(v) for v in losses)
            and abs(losses[0]-losses[1])<.05
            and metrics['gradient_cosine']>=.99 and metrics['relative_gradient_L2_error']<=.15
            and (qk is None or qk>=.95))
        checks.append(metrics)
    model.precision='bf16'
    return {'passed':all(row['passed'] for row in checks),'checks':checks,
        'limits':{'loss_absolute_error':.05,'global_cosine_min':.99,'relative_L2_max':.15,'query_key_cosine_min':.95},
        'scope':'Three real TRAIN probes, two1600-token chunks and one short case; dropout disabled for precision comparison; not a universal numerical guarantee'}


def pilot(config_path,device):
    config=strict_json(config_path);training.verify_bindings(config)
    output=Path(config['output'])/'pilot';output.mkdir(parents=True,exist_ok=False)
    data=CaseData(config['data_manifest']);model=build(config,device)
    atomic_json(output/'trainability.json',trainability(model))
    model.precision='fp32'
    original=training.gradient_gate(model,data,config)
    atomic_json(output/'original_gradient_gate.json',original)
    if not original['passed']:raise ValueError('Original-vs-efficient FP32 gradient gate failed')
    precision=precision_gate(model,data);atomic_json(output/'BF16_gradient_gate.json',precision)
    if not precision['passed']:raise ValueError('BF16 full-context gradients failed the announced gate; no production')
    # This uses the existing replay implementation directly, even though the
    # production fast path avoids replay for small cases.
    with training_functions():
        training.backward_case=replay_backward
        replay=training.replay_gate(model,data.chunks('train',0)[:2],data.rows['train'][0]['label'],1)
    atomic_json(output/'replay_gate.json',replay)
    if not replay['passed']:raise ValueError('BF16 stochastic replay disagrees with joint autograd')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    frozen=frozen_parameter_sha256(model)
    order=np.random.default_rng(config['seed']).permutation(len(data.rows['train'])).tolist()[:16]
    timing=[]
    for begin in range(0,len(order),4):
        selected=order[begin:begin+4];model.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.perf_counter()
        values=[backward_case(model,data.chunks('train',i),data.rows['train'][i]['label'],len(selected),
            config['chunk_microbatch']) for i in selected]
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
            config['clip_norm'],error_if_nonfinite=True))
        if norm>config['maximum_gradient_norm']:raise ValueError('Pilot gradient safety limit exceeded')
        optimizer.step();torch.cuda.synchronize()
        timing.append({'seconds':time.perf_counter()-tick,'cases':len(selected),
            'chunks':sum(row['chunks'] for row in values),'gradient_norm':norm})
    if frozen_parameter_sha256(model)!=frozen:raise ValueError('Frozen backbone changed')
    digest=file_sha256(Path(config_path))
    training.save(output,model,optimizer,{'epoch':0,'next_batch':0,'updates':4},digest,True)
    with training_functions():
        recovery=training.checkpoint_next_update_gate(output,model,optimizer,digest,
            [data.chunks('train',0)[0][:64]],data.rows['train'][0]['label'])
    atomic_json(output/'recovery_gate.json',recovery)
    if not recovery['passed']:raise ValueError('BF16 exact optimizer/RNG recovery gate failed')
    seconds=sum(row['seconds'] for row in timing);chunks=sum(row['chunks'] for row in timing)
    record={'passed':True,'original_gradient_gate':original,'BF16_gradient_gate':precision,
        'replay_gate':replay,'checkpoint_next_update_exact':True,'timing':timing,
        'seconds_per_chunk':seconds/chunks,
        'estimated_train_seconds_per_epoch':seconds/chunks*data.manifest['splits']['train']['chunks'],
        'estimate_caveat':'Only16 complete TRAIN cases; excludes validation, serialization and inference; not a deadline guarantee',
        'peak_allocated_GPU_bytes':torch.cuda.max_memory_allocated(),
        'production_starts_from_own_Phase1_not_pilot':True,'KD_actual_teacher_check_pending':config['method']=='knowledge_distillation'}
    atomic_json(output/'completed.json',record)
    return record


def train(config_path,device,resume=False,stop_after_epoch=None):
    with training_functions():return training.train(config_path,device,resume,stop_after_epoch)


def cache_teacher(campaign_path,device):
    campaign=strict_json(Path(campaign_path)/'campaign.json')
    teacher_file=Path(campaign['configs']['mha_full']['path'])
    student=strict_json(Path(campaign['configs']['mqa_kd']['path']))
    config=strict_json(teacher_file);training.verify_bindings(config)
    folder=Path(config['output']);completed=strict_json(folder/'production/completed.json')
    model=build(config,device)
    training.load(folder/'production/best.pt',model,None,file_sha256(teacher_file),restore=False)
    model.eval().requires_grad_(False);data=CaseData(config['data_manifest'])
    directory=Path(campaign_path)/'teacher';directory.mkdir(exist_ok=False)
    values=[]
    with torch.inference_mode():
        for index,row in enumerate(data.rows['train']):
            logits=predict_case(model,data.chunks('train',index),config['chunk_microbatch'])
            values.append(torch.softmax(logits/student['temperature'],-1).cpu().numpy())
            if index%100==0:atomic_json(directory/'progress.json',{'train_rows_done':index+1,
                'total':len(data.rows['train']),'teacher_frozen':True,'test_labels_used':False})
    path=directory/'probabilities.npy'
    with path.open('xb') as stream:np.save(stream,np.stack(values).astype(np.float32),allow_pickle=False)
    teacher_size=campaign['backbones']['mha']['parameter_count']
    student_size=campaign['backbones']['mqa']['parameter_count']
    if teacher_size<=student_size:raise ValueError('Teacher must be larger than student')
    result={'teacher':'mha_full','teacher_dev_macro_f1':completed['best_dev_macro_f1'],
        'probabilities':str(path),'temperature':student['temperature'],
        'train_ids':[row['id'] for row in data.rows['train']],
        'train_row_keys':[row['row_key'] for row in data.rows['train']],
        'data_manifest_sha256':file_sha256(Path(config['data_manifest'])),
        'teacher_backbone_parameters':teacher_size,'student_backbone_parameters':student_size,
        'partition':'train','teacher_frozen':True,'dev_or_test_targets_generated':False,
        'bindings':{str(path):file_sha256(path),str(teacher_file):file_sha256(teacher_file),
            str(folder/'production/best.pt'):completed['checkpoint_sha256']}}
    atomic_json(directory/'manifest.json',result)
    return result
