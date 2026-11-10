#!/usr/bin/env python3
"""Local evaluator for Phase 4.

Truth JSON: a list of objects, [{"id":"...", "gold":0|1}, ...]
Prediction JSON: [{"id":"...", "pred":0|1}, ...]
"""
import argparse,json,sys
from sklearn.metrics import f1_score,precision_score,recall_score

def read_truth(path):
    try:
        x=json.load(open(path,encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Truth must be a JSON file containing a list of records: {e}")
    if not isinstance(x,list) or not x: raise ValueError('Truth must be a non-empty JSON list.')
    out={}
    for n,r in enumerate(x):
        if not isinstance(r,dict) or 'id' not in r or 'gold' not in r: raise ValueError(f'Truth item {n} must contain id and gold.')
        if not isinstance(r['id'],str) or not r['id'].strip() or isinstance(r['gold'],bool) or r['gold'] not in (0,1): raise ValueError(f'Invalid truth item {n}.')
        if r['id'] in out: raise ValueError(f'Duplicate truth ID: {r["id"]}')
        out[r['id']]=r['gold']
    return out

def read_pred(path):
    x=json.load(open(path,encoding='utf-8'))
    if not isinstance(x,list) or not x: raise ValueError('Prediction must be a non-empty JSON list.')
    out={}
    for n,r in enumerate(x):
        if not isinstance(r,dict) or 'id' not in r or 'pred' not in r: raise ValueError(f'Prediction item {n} must contain id and pred.')
        if not isinstance(r['id'],str) or isinstance(r['pred'],bool) or r['pred'] not in (0,1): raise ValueError(f'Invalid prediction item {n}.')
        if r['id'] in out: raise ValueError(f'Duplicate prediction ID: {r["id"]}')
        out[r['id']]=r['pred']
    return out

def main():
    ap=argparse.ArgumentParser(description='Evaluate Phase 4 binary predictions on your own held-out split.')
    ap.add_argument('--truth',required=True); ap.add_argument('--prediction',required=True); a=ap.parse_args()
    t=read_truth(a.truth); p=read_pred(a.prediction)
    if set(t)!=set(p): raise ValueError(f"ID mismatch: missing={len(set(t)-set(p))}, extra={len(set(p)-set(t))}")
    yt=[t[i] for i in t]; yp=[p[i] for i in t]
    print(f'Examples: {len(t)}')
    print(f'Macro-F1: {f1_score(yt,yp,average="macro",labels=[0,1],zero_division=0):.6f}')
    print(f'Precision: {precision_score(yt,yp,average="binary",pos_label=1,zero_division=0):.6f}')
    print(f'Recall: {recall_score(yt,yp,average="binary",pos_label=1,zero_division=0):.6f}')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'EVALUATION FAILED: {e}',file=sys.stderr); sys.exit(1)
