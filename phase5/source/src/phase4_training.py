"""One declared CJPE experiment: case-normalized loss, full-document inference."""

import datetime
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase2_adaptation import frozen_parameter_sha256
from src.phase3_model import load_backbone
from src.phase3_rebuilt_model import load_rebuilt_backbone
from src.phase4_data import CaseData, package_submission, strict_json
from src.phase4_model import (CaseClassifier, backward_case, batches, case_loss,
    predict_case, trainability, rng_snapshot, rng_restore)

ROOT = Path(__file__).resolve().parents[1]


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def verify_bindings(config):
    for path, digest in config['bindings'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f'Frozen campaign input/code changed: {path}')


def build(config, device):
    architecture = config.get('architecture', 'original')
    if architecture in ('dense', 'encoder_decoder'):
        backbone = load_rebuilt_backbone(config['checkpoint'], config['model_config'], config['checkpoint_sha256'])
    else:
        backbone = load_backbone(config['checkpoint'], config['model_config'])
    model = CaseClassifier(backbone, config['method'], config['head'],
        config['partial_layers'], config['rank'], config['alpha'], config['head_dropout'], architecture)
    return model.to(device)


def rng(cuda=True):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
        'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(0) if cuda else None}


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['cpu'])
    if saved['cuda'] is not None:
        torch.cuda.set_rng_state(saved['cuda'], 0)


def save(output, model, optimizer, progress, digest, improved):
    output = Path(output)
    candidate = output/'checkpoint.partial.pt'
    if candidate.exists():
        raise FileExistsError('Preserve interrupted checkpoint transaction for inspection')
    with candidate.open('xb') as stream:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'progress':progress,'rng':rng(next(model.parameters()).is_cuda),'config_sha256':digest},stream)
        stream.flush()
        os.fsync(stream.fileno())
    candidate.replace(output/'latest.pt')
    checksum = file_sha256(output/'latest.pt')
    atomic_json(output/'latest.metadata.json',{'sha256':checksum,'progress':progress,'config_sha256':digest})
    if improved:
        link = output/'.best.link'
        if link.exists():
            raise FileExistsError('Preserve interrupted best-checkpoint promotion')
        os.link(output/'latest.pt',link)
        os.replace(link,output/'best.pt')
        atomic_json(output/'best.metadata.json',{'sha256':checksum,'progress':progress,'config_sha256':digest})


def load(path, model, optimizer, digest, restore=True):
    metadata = strict_json(Path(path).with_suffix('.metadata.json'))
    if file_sha256(Path(path)) != metadata['sha256'] or metadata['config_sha256'] != digest:
        raise ValueError('Checkpoint hash/config mismatch')
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if saved['config_sha256'] != digest or saved['progress'] != metadata['progress']:
        raise ValueError('Checkpoint payload/metadata mismatch')
    model.load_state_dict(saved['model'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    if restore:
        restore_rng(saved['rng'])
    return saved['progress']


def teacher_targets(config, data):
    if config['method'] != 'knowledge_distillation':
        return None
    path = Path(config['teacher_manifest'])
    record = strict_json(path)
    for filename, digest in record['bindings'].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError('Teacher cache identity changed')
    ids = [row['id'] for row in data.rows['train']]
    if (record['train_ids'] != ids
            or record['train_row_keys'] != [row['row_key'] for row in data.rows['train']]
            or record['temperature'] != config['temperature']
            or record['data_manifest_sha256'] != file_sha256(Path(config['data_manifest']))
            or record['teacher_backbone_parameters'] <= record['student_backbone_parameters']):
        raise ValueError('Teacher/student training alignment or size condition failed')
    values = np.load(record['probabilities'],allow_pickle=False)
    if values.shape != (len(ids),2) or not np.isfinite(values).all() or not np.allclose(values.sum(1),1.,atol=1e-5):
        raise ValueError('Teacher probabilities are incomplete or invalid')
    return values


@torch.no_grad()
def predictions(model, data, split, config):
    model.eval()
    result, logits = [], []
    for index,row in enumerate(data.rows[split]):
        value = predict_case(model,data.chunks(split,index),config['chunk_microbatch']).float().cpu()
        result.append({'id':row['id'],'pred':int(value.argmax())})
        logits.append(value)
    return result,torch.stack(logits)


def evaluate(model,data,config,output,epoch):
    predicted,logits = predictions(model,data,'dev',config)
    gold = [row['label'] for row in data.rows['dev']]
    values = [row['pred'] for row in predicted]
    metrics = {'epoch':epoch,'macro_f1':float(f1_score(gold,values,labels=[0,1],average='macro',zero_division=0)),
        'confusion_matrix':confusion_matrix(gold,values,labels=[0,1]).tolist(),
        'classification_report':classification_report(gold,values,labels=[0,1],output_dict=True,zero_division=0),
        'cases':len(gold),'all_chunks_used':True}
    directory=Path(output)/f'dev_epoch{epoch:03d}'
    if directory.exists():
        directory=Path(output)/f'dev_epoch{epoch:03d}_recovery_{time.time_ns()}'
    directory.mkdir(exist_ok=False)
    atomic_json(directory/'truth.json',[{'id':row['row_key'],'gold':row['label']} for row in data.rows['dev']])
    atomic_json(directory/'prediction.json',[{'id':row['row_key'],'pred':pred['pred']}
        for row,pred in zip(data.rows['dev'],predicted)])
    atomic_json(directory/'row_to_original_id.json',[{'row_key':row['row_key'],'id':row['id']}
        for row in data.rows['dev']])
    metrics['dev_ID_policy']='Stable row keys for evaluator only; all official duplicate rows retained with original-ID mapping'
    result=subprocess.run([sys.executable,str(ROOT/'official/evaluation_scripts/evaluate_phase4.py'),
        '--truth',str(directory/'truth.json'),'--prediction',str(directory/'prediction.json')],
        capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    with (directory/'official_evaluator.txt').open('x') as stream:
        stream.write(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Unchanged official evaluator failed')
    score=float(next(line.split(':')[1] for line in result.stdout.splitlines() if line.startswith('Macro-F1:')))
    if abs(score-metrics['macro_f1'])>5.1e-7:
        raise ValueError('Local metrics disagree with the official evaluator')
    atomic_json(directory/'metrics.json',metrics)
    return metrics


def gradient_gate(model,data,config):
    """FP32 efficient forward/backward versus original A3 manual attention."""
    from src.phase3_model import decoder_hidden
    device=next(model.parameters()).device
    ids,_=next(batches([data.chunks('train',0)[0][:64]],1,device))
    model.eval()
    parameters=[p for p in model.parameters() if p.requires_grad]
    outputs=[]
    gradients=[]
    for manual in (True,False):
        if manual and model.architecture == 'encoder_decoder':
            memory=model.backbone.encode(ids)
            x=model.backbone.tok_emb(ids)+model.backbone.pos_emb[:ids.shape[1]].unsqueeze(0)
            for block in model.backbone.decoder:
                x=block(x,memory)
            hidden=model.backbone.decoder_norm(x)
        elif manual:
            x=model.backbone.tok_emb(ids)+model.backbone.pos_emb[:ids.shape[1]].unsqueeze(0)
            for block in model.backbone.blocks:
                x=block(x)
            hidden=model.backbone.ln_f(x)
        elif model.architecture == 'encoder_decoder':
            from src.phase3_rebuilt_model import hidden_states
            hidden=hidden_states(model.backbone,model.architecture,ids)
        else:
            hidden=decoder_hidden(model.backbone,ids)
        logits=model.classifier(hidden[:,-1])
        loss=case_loss(logits,data.rows['train'][0]['label'])[0]
        gradients.append([g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu')
            for g,p in zip(torch.autograd.grad(loss,parameters,allow_unused=True),parameters)])
        outputs.append(float(loss.detach()))
    a2=b2=dot=error2=0.
    for reference,actual in zip(*gradients):
        reference=reference.double().flatten()
        actual=actual.double().flatten()
        a2+=float(reference.dot(reference));b2+=float(actual.dot(actual))
        dot+=float(reference.dot(actual));error2+=float((reference-actual).square().sum())
    cosine=dot/math.sqrt(a2*b2) if a2 and b2 else 1.
    relative=math.sqrt(error2/max(a2,1e-30))
    passed=(abs(outputs[0]-outputs[1])<1e-4 and cosine>.999 and relative<.01
        and all(math.isfinite(v) for v in [*outputs,cosine,relative]))
    if model.architecture == 'encoder_decoder':
        named=[name for name,p in model.named_parameters() if p.requires_grad]
        required=['backbone.encoder.','backbone.decoder.', '.cross_attn.']
        used={prefix:any(prefix in name and bool(g.abs().sum()>0)
            for name,g in zip(named,gradients[1])) for prefix in required}
        passed=passed and all(used.values())
    else:
        used={}
    return {'passed':passed,'manual_loss':outputs[0],'efficient_loss':outputs[1],
        'gradient_cosine':cosine,'relative_gradient_L2_error':relative,
        'encoder_decoder_nonzero_gradient_groups':used,
        'precision':'FP32, TF32 disabled','scope':'One actual-case diagnostic prefix of at most64 tokens; not a full-context numerical census'}


def replay_gate(model,chunks,label,microbatch=2):
    """Compare real-device dropout replay gradients to a joint autograd graph."""
    device=next(model.parameters()).device
    parameters=[p for p in model.parameters() if p.requires_grad]
    model.train();model.zero_grad(set_to_none=True)
    before=rng_snapshot(device)
    vectors=[model.chunk_vectors(ids,lengths) for ids,lengths in batches(chunks,microbatch,device)]
    logits=model.classifier(torch.cat(vectors).mean(0))
    loss=case_loss(logits,label)[0]
    reference=[value.detach().cpu() if value is not None else torch.zeros_like(parameter,device='cpu')
        for value,parameter in zip(torch.autograd.grad(loss,parameters,allow_unused=True),parameters)]
    expected_rng=rng_snapshot(device)
    expected_loss=float(loss.detach())
    del vectors,logits,loss
    rng_restore(before,device)
    row=backward_case(model,chunks,label,1,microbatch)
    a2=b2=dot=error2=0.
    for expected,parameter in zip(reference,parameters):
        actual=(parameter.grad.detach().cpu() if parameter.grad is not None else torch.zeros_like(expected))
        expected=expected.double().flatten();actual=actual.double().flatten()
        a2+=float(expected.dot(expected));b2+=float(actual.dot(actual))
        dot+=float(expected.dot(actual));error2+=float((expected-actual).square().sum())
    cosine=dot/math.sqrt(a2*b2) if a2 and b2 else 1.
    relative=math.sqrt(error2/max(a2,1e-30))
    actual_rng=rng_snapshot(device)
    rng_matches=torch.equal(expected_rng[0],actual_rng[0]) and (
        expected_rng[1] is None or torch.equal(expected_rng[1],actual_rng[1]))
    passed=(abs(row['loss']-expected_loss)<1e-4 and cosine>.999 and relative<.01
        and rng_matches and all(math.isfinite(v) for v in [cosine,relative,row['loss']]))
    model.zero_grad(set_to_none=True)
    return {'passed':bool(passed),'gradient_cosine':cosine,'relative_L2_error':relative,
        'dropout_RNG_matches_joint_forward':bool(rng_matches),'chunks':len(chunks),
        'loss_error':row['loss']-expected_loss,'joint_gradient_norm':math.sqrt(a2)}


def pilot(config_path,device):
    config=strict_json(config_path)
    verify_bindings(config)
    output=Path(config['output'])/'pilot'
    output.mkdir(parents=True,exist_ok=False)
    data=CaseData(config['data_manifest'])
    model=build(config,device)
    atomic_json(output/'trainability.json',trainability(model))
    gate=gradient_gate(model,data,config)
    atomic_json(output/'gradient_gate.json',gate)
    if not gate['passed']:
        raise ValueError('Manual/efficient gradient gate failed; no optimizer updates')
    chunks=data.chunks('train',0)[:2]  # Diagnostic only; production never caps cases.
    replay=replay_gate(model,chunks,data.rows['train'][0]['label'],config['chunk_microbatch'])
    atomic_json(output/'replay_gate.json',replay)
    if not replay['passed']:
        raise ValueError('Real-device dropout gradient replay disagrees with joint autograd; no updates')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    frozen=frozen_parameter_sha256(model)
    started=time.perf_counter()
    losses=[]
    for _ in range(4):
        model.train();optimizer.zero_grad(set_to_none=True)
        row=backward_case(model,chunks,data.rows['train'][0]['label'],1,config['chunk_microbatch'])
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        if norm>config['maximum_gradient_norm']:
            raise ValueError('Large finite gradient failed pilot safety gate')
        optimizer.step();losses.append({**row,'gradient_norm':norm})
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    if frozen_parameter_sha256(model)!=frozen:
        raise ValueError('Frozen weights changed')
    state={'epoch':0,'next_batch':0,'updates':4,'best_score':None,'bad_epochs':0}
    digest=file_sha256(Path(config_path))
    save(output,model,optimizer,state,digest,True)
    recovery=checkpoint_next_update_gate(output,model,optimizer,digest,
        [chunks[0][:64]],data.rows['train'][0]['label'])
    atomic_json(output/'checkpoint_recovery_gate.json',recovery)
    if not recovery['passed']:
        raise ValueError('Restored optimizer/model/RNG does not reproduce next update exactly')
    estimate=seconds/(4*len(chunks))*data.manifest['splits']['train']['chunks']
    record={'passed':True,'gradient_gate':gate,'replay_gate':replay,'updates':4,'losses':losses,
        'actual_full_context_chunks_in_pilot':len(chunks),'pilot_case_chunk_cap':2,
        'production_chunk_cap':None,'seconds':seconds,'estimated_train_seconds_per_epoch':estimate,
        'estimate_caveat':'Small diagnostic sample; excludes full dev/test inference and I/O, not a completion guarantee',
        'peak_allocated_GPU_bytes':torch.cuda.max_memory_allocated(),
        'frozen_parameters_unchanged':True,'checkpoint_restore_passed':True,
        'checkpoint_next_update_exact':recovery['passed'],
        'KD_actual_teacher_check_pending':config['method']=='knowledge_distillation',
        'full_training_started':False}
    atomic_json(output/'completed.json',record)
    return record


def checkpoint_next_update_gate(output,model,optimizer,digest,chunks,label):
    """One identical stochastic update, first uninterrupted and then restored."""
    from src.phase1_full_training import state_digest
    def update():
        model.train();optimizer.zero_grad(set_to_none=True)
        value=backward_case(model,chunks,label,1,1)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
            1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        return value
    load(Path(output)/'latest.pt',model,optimizer,digest)
    first=update()
    expected=(state_digest(model.state_dict()),state_digest(optimizer.state_dict()))
    expected_rng=rng(next(model.parameters()).is_cuda)
    load(Path(output)/'latest.pt',model,optimizer,digest)
    second=update()
    actual=(state_digest(model.state_dict()),state_digest(optimizer.state_dict()))
    actual_rng=rng(next(model.parameters()).is_cuda)
    same_rng=(torch.equal(expected_rng['cpu'],actual_rng['cpu'])
        and (expected_rng['cuda'] is None or torch.equal(expected_rng['cuda'],actual_rng['cuda'])))
    load(Path(output)/'latest.pt',model,optimizer,digest)
    return {'passed':expected==actual and same_rng and first==second,
        'weights_exact':expected[0]==actual[0],'optimizer_exact':expected[1]==actual[1],
        'dropout_rng_exact':same_rng,'loss_exact':first==second,
        'scope':'One actual-case prefix at most64 tokens on the assigned device; production checkpoint restored afterward'}


def schedule_settings(config, training_rows):
    """The schedule is immutable; the next review/stop epoch is not its horizon."""
    updates_per_epoch=math.ceil(training_rows/config['cases_per_update'])
    kind=config.get('lr_schedule','cosine')
    if kind not in ('cosine','warmup_constant'):
        raise ValueError('Unknown learning-rate schedule')
    total=updates_per_epoch*config['max_epochs']
    warmup=config.get('warmup_updates',max(1,int(total*config['warmup_fraction'])))
    if type(warmup) is not int or warmup<1:
        raise ValueError('Warmup must be a positive, fixed update count')
    return {'kind':kind,'warmup_updates':warmup,'updates_per_epoch':updates_per_epoch,
        'cosine_total_updates':total if kind=='cosine' else None,
        'backbone_peak':config['learning_rate'],'head_peak':config['head_learning_rate']}


def schedule_factor(schedule, step):
    if step<1:raise ValueError('Optimizer updates are one-based')
    warmup=schedule['warmup_updates']
    if step<=warmup:return step/warmup
    if schedule['kind']=='warmup_constant':return 1.
    return .1+.9*.5*(1+math.cos(math.pi*min(1.,
        (step-warmup)/max(1,schedule['cosine_total_updates']-warmup))))


def train(config_path,device,resume=False,stop_after_epoch=None):
    config=strict_json(config_path)
    verify_bindings(config)
    campaign=Path(config['campaign'])
    approval=strict_json(Path(config.get('pilot_approval_path',campaign/'pilot_approval.json')))
    if approval['campaign_sha256']!=file_sha256(campaign/'campaign.json'):
        raise ValueError('Pilot approval belongs to a different campaign')
    own_pilot=strict_json(Path(config['output'])/'pilot/completed.json')
    if not own_pilot['passed'] or approval['pilot_hashes'][config['name']] != file_sha256(Path(config['output'])/'pilot/completed.json'):
        raise ValueError('Pilot must pass before full training')
    output=Path(config['output'])/'production'
    if output.exists() and not resume:
        raise FileExistsError('Existing production attempt must be explicitly resumed, not overwritten')
    output.mkdir(exist_ok=resume)
    data=CaseData(config['data_manifest'])
    schedule=schedule_settings(config,len(data.rows['train']))
    limit=config['max_epochs'] if stop_after_epoch is None else stop_after_epoch
    if type(limit) is not int or limit<1:
        raise ValueError('Stop epoch must be a positive integer')
    if limit>config['max_epochs'] and (not resume or schedule['kind']!='warmup_constant'):
        raise ValueError('Extension requires explicit resume and an endpoint-independent schedule')
    model=build(config,device)
    summary=trainability(model)
    atomic_json(output/'trainability.json',summary)
    frozen=frozen_parameter_sha256(model)
    teacher=teacher_targets(config,data)
    if teacher is not None:
        atomic_json(output/'KD_identity.json',{'manifest':config['teacher_manifest'],
            'sha256':file_sha256(Path(config['teacher_manifest'])),'training_cases':len(teacher),
            'equation':'0.5 CE +0.5 T^2 KL(teacher_T || student_T)','temperature':config['temperature']})
    groups=[{'params':[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('backbone.')],
        'lr':config['learning_rate'],'peak':config['learning_rate']},
        {'params':list(model.classifier.parameters()),'lr':config['head_learning_rate'],'peak':config['head_learning_rate']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    digest=file_sha256(Path(config_path))
    progress={'epoch':0,'next_batch':0,'updates':0,'best_score':None,'bad_epochs':0}
    if resume:
        progress=load(output/'latest.pt',model,optimizer,digest)
        if progress['epoch']>limit:
            raise ValueError('Cannot resume to an earlier epoch')
        if 'schedule' in progress and progress['schedule']!=schedule:
            raise ValueError('Checkpoint learning-rate schedule changed')
    progress['schedule']=schedule
    atomic_json(output/f'train_request_{time.time_ns()}.json',{'resume':resume,
        'stop_after_epoch':limit,'config_sha256':digest,'schedule':schedule,
        'resume_from':'latest.pt' if resume else None})
    started=time.perf_counter()
    log=output/('updates_resume_'+str(time.time_ns())+'.jsonl' if resume else 'updates.jsonl')
    for epoch in range(progress['epoch'],limit):
        order=np.random.default_rng(config['seed']+epoch).permutation(len(data.rows['train'])).tolist()
        batches_of_cases=[order[i:i+config['cases_per_update']] for i in range(0,len(order),config['cases_per_update'])]
        for batch_number in range(progress['next_batch'],len(batches_of_cases)):
            indices=batches_of_cases[batch_number]
            step=progress['updates']+1
            factor=schedule_factor(schedule,step)
            for group in optimizer.param_groups:group['lr']=group['peak']*factor
            optimizer.zero_grad(set_to_none=True);model.train()
            values=[];tick=time.perf_counter()
            for index in indices:
                row=data.rows['train'][index]
                values.append(backward_case(model,data.chunks('train',index),row['label'],len(indices),
                    config['chunk_microbatch'],None if teacher is None else teacher[index],
                    config['temperature'],config['kd_weight']))
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                config['clip_norm'],error_if_nonfinite=True))
            if norm>config['maximum_gradient_norm']:
                raise ValueError(f'Large finite gradient {norm}; stop rather than silently training through instability')
            optimizer.step()
            progress.update(epoch=epoch,next_batch=batch_number+1,updates=step)
            append(log,{'epoch':epoch+1,'batch':batch_number+1,'updates':step,
                'cases':len(indices),'chunks':sum(v['chunks'] for v in values),
                'mean_case_loss':sum(v['loss'] for v in values)/len(indices),
                'gradient_norm':norm,'clipped':norm>config['clip_norm'],
                'backbone_learning_rate':optimizer.param_groups[0]['lr'],
                'head_learning_rate':optimizer.param_groups[1]['lr'],
                'seconds':time.perf_counter()-tick})
            if step%config['checkpoint_every_updates']==0:
                save(output,model,optimizer,progress,digest,False)
                atomic_json(output/'progress.json',{**progress,'elapsed_seconds':time.perf_counter()-started})
        metrics=evaluate(model,data,config,output,epoch+1)
        improved=progress['best_score'] is None or metrics['macro_f1']>progress['best_score']
        if improved:
            progress['best_score']=metrics['macro_f1'];progress['bad_epochs']=0
        else:progress['bad_epochs']+=1
        progress.update(epoch=epoch+1,next_batch=0)
        if frozen_parameter_sha256(model)!=frozen:
            raise ValueError('Frozen backbone parameters changed')
        save(output,model,optimizer,progress,digest,improved)
        atomic_json(output/'progress.json',{**progress,'latest_dev':metrics})
        if (not config.get('disable_early_stopping',False)
                and epoch+1>=config['min_epochs'] and progress['bad_epochs']>=config['patience']):break
    load(output/'best.pt',model,None,digest,restore=False)
    test_predictions,_=predictions(model,data,'test',config)
    export=(output/'exports'/f'epoch{progress["epoch"]:03d}'
        if schedule['kind']=='warmup_constant' else output)
    export.mkdir(parents=True,exist_ok=True)
    atomic_json(export/'prediction.json',test_predictions)
    archive_path=export/'submission.zip'
    if archive_path.exists() and resume:
        import zipfile
        from src.phase4_data import validate_predictions
        with zipfile.ZipFile(archive_path) as existing:
            if existing.namelist()!=['prediction.json'] or existing.testzip() is not None:
                raise ValueError('Existing interrupted-finalization ZIP is invalid')
            old_predictions=json.loads(existing.read('prediction.json'))
        validate_predictions(old_predictions,[r['id'] for r in data.rows['test']])
        if old_predictions!=test_predictions:
            raise ValueError('Existing ZIP does not reproduce the selected best checkpoint')
        archive={'path':str(archive_path),'sha256':file_sha256(archive_path),'predictions':len(test_predictions)}
    else:
        archive=package_submission(archive_path,test_predictions,[r['id'] for r in data.rows['test']])
    verify_bindings(config)
    result={'status':'completed','best_dev_macro_f1':progress['best_score'],'epochs':progress['epoch'],
        'updates':progress['updates'],'checkpoint':str(output/'best.pt'),
        'checkpoint_sha256':file_sha256(output/'best.pt'),'submission':archive,
        'elapsed_seconds':time.perf_counter()-started,'all_cases_all_chunks':True,
        'frozen_parameters_unchanged':True,'test_labels_used':False,'uploaded':False}
    if export!=output:atomic_json(export/'completed.json',result)
    atomic_json(output/'completed.json',result)
    return result
