#!/usr/bin/env python3
"""Local evaluator for Phase 2.

Usage:
  python evaluate_phase2.py --truth truth.json --prediction prediction.json --labels labels.txt
"""
import argparse, json, sys

class DuplicateJSONKeyError(ValueError): pass
def reject_constant(v): raise ValueError(f"Non-standard JSON numeric value '{v}' is not allowed.")
def pairs_no_duplicates(pairs):
    out={}; dup=[]
    for k,v in pairs:
        if k in out: dup.append(k)
        out[k]=v
    if dup: raise DuplicateJSONKeyError(f"Duplicate JSON object keys detected: {list(dict.fromkeys(dup))[:10]}.")
    return out
def load_json(path):
    with open(path,encoding='utf-8') as f:
        try: return json.load(f, object_pairs_hook=pairs_no_duplicates, parse_constant=reject_constant)
        except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON in {path} at line {e.lineno}, column {e.colno}: {e.msg}") from e
def labels(path):
    with open(path,encoding='utf-8') as f: xs=[x.strip() for x in f if x.strip()]
    if not xs: raise ValueError('labels.txt is empty.')
    if len(xs)!=12: raise ValueError(f'Expected 12 labels, found {len(xs)}.')
    if len(set(xs))!=len(xs): raise ValueError('labels.txt contains duplicate labels.')
    return xs
def spans(x, allowed, where):
    if x is None: return set()
    if not isinstance(x,list): raise ValueError(f'{where}: expected a list of entities.')
    out=set()
    for j,s in enumerate(x):
        if not isinstance(s,list) or len(s) not in (3,4): raise ValueError(f'{where}, entity {j}: expected [start,end,label] or [start,end,label,entity_text].')
        a,b,l=s[:3]
        if isinstance(a,bool) or not isinstance(a,int) or isinstance(b,bool) or not isinstance(b,int): raise ValueError(f'{where}, entity {j}: start/end must be integers.')
        if a<0 or b<=a: raise ValueError(f'{where}, entity {j}: require 0 <= start < end.')
        if not isinstance(l,str) or l not in allowed: raise ValueError(f'{where}, entity {j}: invalid label {l!r}.')
        if len(s)==4 and not isinstance(s[3],str): raise ValueError(f'{where}, entity {j}: entity_text must be a string.')
        t=(a,b,l)
        if t in out: raise ValueError(f'{where}, entity {j}: duplicate span {t}.')
        out.add(t)
    return out

def main():
    ap=argparse.ArgumentParser(description='Evaluate Phase 2 entity predictions on your own held-out split.')
    ap.add_argument('--truth',required=True); ap.add_argument('--prediction',required=True); ap.add_argument('--labels',required=True)
    args=ap.parse_args(); truth=load_json(args.truth); pred=load_json(args.prediction); labs=labels(args.labels)
    if not isinstance(truth,dict) or not truth or not isinstance(pred,dict) or not pred: raise ValueError('Truth and prediction must be non-empty JSON objects.')
    if set(truth)!=set(pred): raise ValueError(f"ID mismatch: missing={len(set(truth)-set(pred))}, extra={len(set(pred)-set(truth))}")
    stats={l:[0,0,0] for l in labs}; allowed=set(labs)
    for did in truth:
        g=spans(truth[did],allowed,f'truth[{did}]'); p=spans(pred[did],allowed,f'prediction[{did}]')
        for l in labs:
            gs={x for x in g if x[2]==l}; ps={x for x in p if x[2]==l}
            stats[l][0]+=len(gs&ps); stats[l][1]+=len(ps-gs); stats[l][2]+=len(gs-ps)
    precisions=[]; recalls=[]; f1s=[]
    print(f'Examples: {len(truth)}')
    for l in labs:
        tp,fp,fn=stats[l]; pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f=2*pr*rc/(pr+rc) if pr+rc else 0
        precisions.append(pr); recalls.append(rc); f1s.append(f)
        print(f'{l}: P={pr:.6f} R={rc:.6f} F1={f:.6f}')
    print(f'Strict Macro-F1:        {sum(f1s)/len(f1s):.6f}')
    print(f'Strict Macro-Precision: {sum(precisions)/len(precisions):.6f}')
    print(f'Strict Macro-Recall:    {sum(recalls)/len(recalls):.6f}')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'EVALUATION FAILED: {e}',file=sys.stderr); sys.exit(1)
