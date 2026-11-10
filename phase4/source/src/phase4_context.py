"""Full-case context ablations around the submitted dense LoRMA linear model.

All chunks remain independent causal encoder inputs. Only their document-level
aggregation changes. Larger cases replay exact chunk gradients without keeping
every transformer graph in memory. Existing running campaign files are untouched.
"""

import contextlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.phase1_full_corpus_prepare import atomic_json,file_sha256
from src.phase2_adaptation import frozen_parameter_sha256
from src.phase3_rebuilt_model import load_rebuilt_backbone
from src.phase4_data import CaseData,strict_json,chunk_slices
from src.phase4_deadline import DeadlineClassifier,precision_gate
from src.phase4_model import batches,case_loss,rng_snapshot,rng_restore,trainability
import src.phase4_training as training

KINDS=('mean','token_weighted_mean','max','attention','ordered_gru')


def ranges(length,capacity=1600,layout='full'):
    if layout=='overlap':
        stride=capacity-1-256
        if stride<1:raise ValueError('Overlap must be smaller than content capacity')
        result=[]
        for start in range(0,max(1,length),stride):
            end=min(length,start+capacity-1);result.append((start,end))
            if end==length:break
        return result
    result=chunk_slices(length,capacity)
    if layout=='selected_parts':return [result[i] for i in sorted({0,len(result)//2,len(result)-1})]
    if layout=='ending_only':return result[-1:]
    if layout!='full':raise ValueError('Unknown document layout')
    return result


class ContextData(CaseData):
    def __init__(self,config):
        super().__init__(config['data_manifest'])
        self.config=config
        self.manifest['capacity']=config['capacity']
        for split,entry in self.manifest['splits'].items():
            entry['chunks']=sum(len(ranges(row['tokens'],config['capacity'],config['layout'])) for row in self.rows[split])

    def chunks(self,split,index):
        row=self.rows[split][index];offset,length=row['offset'],row['tokens']
        values=self.tokens[split][offset:offset+length]
        return [[2]+values[a:b].astype(np.int64).tolist()
            for a,b in ranges(length,self.config['capacity'],self.config['layout'])]


class ChunkPool(nn.Module):
    def __init__(self,width,kind):
        super().__init__()
        if kind not in KINDS:raise ValueError('Unknown context aggregation')
        self.kind=kind
        if kind=='attention':
            self.score=nn.Linear(width,1,bias=False)
            nn.init.zeros_(self.score.weight)  # Starts as the working mean pool.
        if kind=='ordered_gru':
            self.gru=nn.GRU(width,128,batch_first=True)
            self.project=nn.Linear(128,width)
            self.gate=nn.Parameter(torch.zeros(()))

    def forward(self,vectors,lengths):
        if vectors.ndim!=2 or len(vectors)==0 or lengths.shape!=(len(vectors),):
            raise ValueError('One ordered vector and length required per chunk')
        if bool((lengths<1).any()):raise ValueError('Chunks must include at least BOS')
        if self.kind=='mean':return vectors.mean(0)
        if self.kind=='token_weighted_mean':
            weights=(lengths-1).clamp_min(1).to(vectors.dtype)
            return (vectors*weights[:,None]).sum(0)/weights.sum()
        if self.kind=='max':return vectors.max(0).values  # Earliest chunk wins exact ties.
        normalized=F.layer_norm(vectors,(vectors.shape[-1],))
        if self.kind=='attention':
            weights=torch.softmax(self.score(normalized).squeeze(-1),0)
            return (vectors*weights[:,None]).sum(0)
        # Zero residual gate preserves the trained mean representation initially.
        # It then learns an ordered, full-sequence correction. No chunk cap.
        _,last=self.gru(normalized.unsqueeze(0))
        return vectors.mean(0)+self.gate.tanh()*self.project(last[-1,0])


class DocumentHead(nn.Module):
    def __init__(self,linear,width,kind,head='linear'):
        super().__init__()
        self.linear=linear
        self.pooling=ChunkPool(width,kind)
        self.residual=None
        if head=='residual_mlp':
            self.residual=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),
                nn.Dropout(.1),nn.Linear(256,2))
            nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        elif head!='linear':raise ValueError('Unknown classification head')

    def forward(self,pooled):
        logits=self.linear(pooled)
        return logits if self.residual is None else logits+self.residual(pooled)


class ContextClassifier(DeadlineClassifier):
    def __init__(self,backbone,kind,precision='bf16',head='linear'):
        super().__init__(backbone,'dense','lorma_plus',precision)
        self.classifier=DocumentHead(self.classifier,backbone.cfg.d_model,kind,head)
        self.context_kind=kind
        self.head_type=head

    def aggregate(self,vectors,lengths):return self.classifier.pooling(vectors.float(),lengths)


def build(config,device):
    if config['architecture']!='dense' or config['method']!='lorma_plus' or config['head'] not in ('linear','residual_mlp'):
        raise ValueError('Keep the selected dense LoRMA backbone and linear head')
    source=Path(config['warm_start_checkpoint'])
    if file_sha256(source)!=config['warm_start_sha256']:raise ValueError('Submitted parent checkpoint changed')
    metadata=strict_json(source.with_suffix('.metadata.json'))
    if metadata['sha256']!=config['warm_start_sha256'] or metadata['config_sha256']!=config['warm_start_config_sha256']:
        raise ValueError('Parent checkpoint metadata/config mismatch')
    base=load_rebuilt_backbone(config['checkpoint'],config['model_config'],config['checkpoint_sha256'])
    model=ContextClassifier(base,config['context_kind'],config['compute_precision'],config['head'])
    model.class_weights=config.get('class_weights')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    if saved['config_sha256']!=config['warm_start_config_sha256'] or saved['progress']['epoch']!=1:
        raise ValueError('Expected the exact archived submitted epoch1 model')
    mapped={('classifier.linear.'+key[len('classifier.'):] if key.startswith('classifier.') else key):value
        for key,value in saved['model'].items()}
    missing,unexpected=model.load_state_dict(mapped,strict=False)
    expected={name for name in model.state_dict() if name.startswith(('classifier.pooling.','classifier.residual.'))}
    if set(missing)!=expected or unexpected:raise ValueError('Warm-start tensor mapping is incomplete')
    actual=model.state_dict()
    if any(not torch.equal(actual[key],value) for key,value in mapped.items()):
        raise ValueError('Warm-start weights must match the submitted parent exactly')
    return model.to(device)


def backward_case(model,chunks,label,denominator,microbatch=4,teacher=None,temperature=2.,kd_weight=.5,class_weights=None,
                  force_replay=False):
    if not model.training or denominator<1:raise ValueError('Training and positive actual case denominator required')
    if class_weights is None:class_weights=getattr(model,'class_weights',None)
    device=next(model.parameters()).device
    if len(chunks)<=microbatch and not force_replay:
        ids,lengths=next(batches(chunks,microbatch,device))
        logits=model.classifier(model.aggregate(model.chunk_vectors(ids,lengths),lengths))
        loss,ce,kl=case_loss(logits,label,teacher,temperature,kd_weight,class_weights)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite case loss')
        (loss/denominator).backward()
    else:
        states=[];values=[];all_lengths=[]
        with torch.no_grad():
            for ids,lengths in batches(chunks,microbatch,device):
                states.append(rng_snapshot(device));all_lengths.append(lengths)
                values.append(model.chunk_vectors(ids,lengths))
        vectors=torch.cat(values).detach().requires_grad_(True)
        logits=model.classifier(model.aggregate(vectors,torch.cat(all_lengths)))
        loss,ce,kl=case_loss(logits,label,teacher,temperature,kd_weight,class_weights)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite case loss')
        (loss/denominator).backward()
        derivative=vectors.grad.detach();after=rng_snapshot(device);offset=0
        try:
            for state,(ids,lengths) in zip(states,batches(chunks,microbatch,device)):
                rng_restore(state,device)
                actual=model.chunk_vectors(ids,lengths)
                actual.backward(derivative[offset:offset+len(actual)])
                offset+=len(actual)
        finally:rng_restore(after,device)
        if offset!=len(chunks):raise ValueError('Gradient replay omitted chunks')
    return {'loss':float(loss.detach()),'ce':float(ce),'kl':float(kl),'chunks':len(chunks)}


@torch.no_grad()
def predict_case(model,chunks,microbatch=4):
    if model.training:raise ValueError('Inference must disable dropout')
    device=next(model.parameters()).device;values=[];all_lengths=[]
    for ids,lengths in batches(chunks,microbatch,device):
        values.append(model.chunk_vectors(ids,lengths));all_lengths.append(lengths)
    vectors=torch.cat(values)
    if len(vectors)!=len(chunks):raise ValueError('Inference omitted document chunks')
    logits=model.classifier(model.aggregate(vectors,torch.cat(all_lengths)))
    if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite case logits')
    return logits


@contextlib.contextmanager
def functions(config=None):
    previous=(training.build,training.backward_case,training.predict_case,training.predictions,training.CaseData)
    original_predictions=training.predictions
    def predictions(model,data,split,config):
        result,logits=original_predictions(model,data,split,config)
        if split=='dev':
            from sklearn.metrics import roc_auc_score
            probability=logits.softmax(-1)[:,1].numpy();gold=[r['label'] for r in data.rows['dev']]
            record={'split':'dev','rows':len(gold),'class1_predictions':int((logits.argmax(-1)==1).sum()),
                'class1_probability_quantiles':np.quantile(probability,[0,.05,.5,.95,1]).tolist(),
                'roc_auc':float(roc_auc_score(gold,probability)),
                'constant_class_prediction':len(set(row['pred'] for row in result))==1,
                'no_test_score_tuning':True}
            atomic_json(Path(config['output'])/'production/dev_learning_quality.json',record)
            np.save(Path(config['output'])/'production/dev_logits.npy',logits.numpy(),allow_pickle=False)
        elif split=='test':
            np.save(Path(config['output'])/'production/test_logits.npy',logits.numpy(),allow_pickle=False)
        return result,logits
    training.build=build;training.backward_case=backward_case
    training.predict_case=predict_case;training.predictions=predictions
    if config is not None:
        def data(path):
            if str(path)!=config['data_manifest']:raise ValueError('Wrong data manifest')
            return ContextData(config)
        training.CaseData=data
    try:yield
    finally:training.build,training.backward_case,training.predict_case,training.predictions,training.CaseData=previous


def replay_gate(model,chunks,label):
    device=next(model.parameters()).device;model.train();model.zero_grad(set_to_none=True)
    # Exercise a nonzero learned GRU residual during the diagnostic, not just its
    # initial mean-pool identity. Restore the gate before any pilot learning.
    pool=model.classifier.pooling;original_gate=None
    if pool.kind=='ordered_gru':
        original_gate=pool.gate.detach().clone()
        with torch.no_grad():pool.gate.fill_(.1)
    try:
        before=rng_snapshot(device);vectors=[];sizes=[]
        for ids,lengths in batches(chunks,1,device):
            vectors.append(model.chunk_vectors(ids,lengths));sizes.append(lengths)
        loss=case_loss(model.classifier(model.aggregate(torch.cat(vectors),torch.cat(sizes))),label)[0]
        parameters=[p for p in model.parameters() if p.requires_grad]
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        reference=[g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu') for p,g in zip(parameters,gradients)]
        expected_loss=float(loss.detach());after=rng_snapshot(device)
        del vectors,loss,gradients
        rng_restore(before,device)
        value=backward_case(model,chunks,label,1,1,force_replay=True)
        aa=bb=dot=error=0.
        for p,g in zip(parameters,reference):
            actual=p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(g)
            g=g.double().flatten();actual=actual.double().flatten()
            aa+=float(g.dot(g));bb+=float(actual.dot(actual));dot+=float(g.dot(actual));error+=float((g-actual).square().sum())
        rng=rng_snapshot(device);same=torch.equal(after[0],rng[0]) and (after[1] is None or torch.equal(after[1],rng[1]))
        cosine=dot/max((aa*bb)**.5,1e-30);relative=(error/max(aa,1e-30))**.5
        return {'passed':abs(value['loss']-expected_loss)<1e-4 and cosine>.999 and relative<.01 and same,
            'cosine':cosine,'relative_L2_error':relative,'exact_dropout_rng':same,'chunks':len(chunks),
            'GRU_nonzero_residual_probe':original_gate is not None}
    finally:
        model.zero_grad(set_to_none=True)
        if original_gate is not None:
            with torch.no_grad():pool.gate.copy_(original_gate)


def pilot(config_path,device):
    config=strict_json(config_path);training.verify_bindings(config)
    output=Path(config['output'])/'pilot';output.mkdir(exist_ok=False)
    data=ContextData(config);model=build(config,device)
    atomic_json(output/'trainability.json',trainability(model))
    model.precision='fp32';manual=training.gradient_gate(model,data,config)
    atomic_json(output/'manual_gate.json',manual)
    if not manual['passed']:raise ValueError('Manual/efficient gradient gate failed')
    precision=precision_gate(model,data)
    precision['scope']='Three TRAIN probes at this variant context capacity; exact token counts recorded in checks; not a full numerical census'
    atomic_json(output/'precision_gate.json',precision)
    if not precision['passed']:raise ValueError('FP32/BF16 gradient gate failed')
    replay=replay_gate(model,data.chunks('train',0)[:2],data.rows['train'][0]['label'])
    atomic_json(output/'replay_gate.json',replay)
    if not replay['passed']:raise ValueError('Context-pool gradient replay failed')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    frozen=frozen_parameter_sha256(model);order=np.random.default_rng(config['seed']).permutation(len(data.rows['train']))[:16]
    timings=[]
    for start in range(0,len(order),4):
        selected=order[start:start+4];model.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.perf_counter()
        values=[backward_case(model,data.chunks('train',int(i)),data.rows['train'][int(i)]['label'],len(selected),
            config['chunk_microbatch']) for i in selected]
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        if norm>config['maximum_gradient_norm']:raise ValueError('Pilot gradient safety limit exceeded')
        optimizer.step();torch.cuda.synchronize()
        timings.append({'seconds':time.perf_counter()-tick,'chunks':sum(v['chunks'] for v in values),'gradient_norm':norm})
    if frozen_parameter_sha256(model)!=frozen:raise ValueError('Frozen Phase1 tensors changed')
    digest=file_sha256(config_path)
    training.save(output,model,optimizer,{'epoch':0,'next_batch':0,'updates':4,'best_score':None,'bad_epochs':0},digest,True)
    with functions(config):
        recovery=training.checkpoint_next_update_gate(output,model,optimizer,digest,[data.chunks('train',0)[0][:64]],data.rows['train'][0]['label'])
    atomic_json(output/'recovery_gate.json',recovery)
    if not recovery['passed']:raise ValueError('Exact optimizer/model/RNG recovery failed')
    estimate=sum(v['seconds'] for v in timings)/sum(v['chunks'] for v in timings)*data.manifest['splits']['train']['chunks']
    record={'passed':True,'estimated_train_seconds_per_epoch':estimate,'timings':timings,
        'all_16_pilot_cases_complete':True,'production_case_cap':None,
        'production_layout':config['layout'],'selected_context_explicit':config['layout'] in ('selected_parts','ending_only'),
        'pilot_weights_discarded':True,'production_starts_from_submitted_1007928':True,
        'estimate_excludes_dev_test_and_IO':True,'manual_gate':manual,'precision_gate':precision,'replay_gate':replay,
        'exact_checkpoint_recovery':recovery['passed']}
    atomic_json(output/'completed.json',record)
    return record


def train(config_path,device,resume=False):
    config=strict_json(config_path)
    with functions(config):return training.train(config_path,device,resume=resume)
