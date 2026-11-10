#!/usr/bin/env python3
"""Local evaluator for Phase 5.

Truth JSON: a list of objects, [{"id":"...", "gold":[0,2,5,...]}, ...]
Prediction JSON: [{"id":"...", "pred":[0,2,5,...]}, ...]
"""
import argparse,json,sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def read_truth(path):
    try:
        x=json.load(open(path,encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Truth must be a JSON file containing a list of records: {e}")
    if not isinstance(x,list) or not x: raise ValueError('Truth must be a non-empty JSON list.')
    out={}; max_label=-1
    for n,r in enumerate(x):
        if not isinstance(r,dict) or 'id' not in r or 'gold' not in r: raise ValueError(f'Truth item {n} must contain id and gold.')
        if not isinstance(r['id'],str) or not r['id'].strip() or not isinstance(r['gold'],list): raise ValueError(f'Invalid truth item {n}.')
        if r['id'] in out: raise ValueError(f'Duplicate truth ID: {r["id"]}')
        seen=set()
        for label in r['gold']:
            if isinstance(label,bool) or not isinstance(label,int) or label<0 or label in seen: raise ValueError(f'Invalid/duplicate label in truth item {n}.')
            seen.add(label); max_label=max(max_label,label)
        out[r['id']]=r['gold']
    return out,max_label+1

def read_pred(path,nlab):
    x=json.load(open(path,encoding='utf-8'))
    if not isinstance(x,list) or not x: raise ValueError('Prediction must be a non-empty JSON list.')
    out={}
    for n,r in enumerate(x):
        if not isinstance(r,dict) or 'id' not in r or 'pred' not in r: raise ValueError(f'Prediction item {n} must contain id and pred.')
        if not isinstance(r['id'],str) or not isinstance(r['pred'],list) or r['id'] in out: raise ValueError(f'Invalid or duplicate prediction item {n}.')
        seen=set()
        for x in r['pred']:
            if isinstance(x,bool) or not isinstance(x,int) or x<0 or x>=nlab or x in seen: raise ValueError(f'Invalid/duplicate label in prediction item {n}.')
            seen.add(x)
        out[r['id']]=r['pred']
    return out

def main():
    ap=argparse.ArgumentParser(description='Evaluate Phase 5 multilabel predictions on your own held-out split.')
    ap.add_argument('--truth',required=True); ap.add_argument('--prediction',required=True); a=ap.parse_args()
    t,n=read_truth(a.truth); p=read_pred(a.prediction,n)
    if set(t)!=set(p): raise ValueError(f"ID mismatch: missing={len(set(t)-set(p))}, extra={len(set(p)-set(t))}")
    yt=np.zeros((len(t),n),dtype=np.uint8); yp=np.zeros((len(t),n),dtype=np.uint8)
    for i,k in enumerate(t):
        for x in t[k]: yt[i,x]=1
        for x in p[k]: yp[i,x]=1
    print(f'Examples: {len(t)}'); print(f'Labels: {n}')
    print(f'Macro-F1: {f1_score(yt,yp,average="macro",zero_division=0):.6f}')
    micro=f1_score(yt,yp,average='micro',zero_division=0)
    subset=accuracy_score(yt,yp)
    print(f'Micro-F1: {micro:.6f}')
    print(f'Subset Accuracy: {subset:.6f}')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'EVALUATION FAILED: {e}',file=sys.stderr); sys.exit(1)
