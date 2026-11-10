#!/usr/bin/env python3
"""Local evaluator for Phase 6.

Requires: rouge-score, bert-score, torch.
Truth JSON: {"doc_id":"reference summary", ...}
Prediction JSON: either {"doc_id":"summary", ...} or [{"id":"...","summary":"..."}, ...]
"""
import argparse,json,sys
try:
    from rouge_score import tokenizers, rouge_scorer
    import bert_score
except ImportError as e:
    print('Missing dependency. Install requirements.txt first.', file=sys.stderr); raise

def load(path):
    with open(path,encoding='utf-8') as f: return json.load(f)

def lcs(a,b):
    if len(a)>len(b): a,b=b,a
    masks={}
    for i,t in enumerate(a): masks[t]=masks.get(t,0)|(1<<i)
    s=0
    for t in b:
        m=masks.get(t,0); x=m|s; y=(s<<1)|1; s=x & ~(x-y)
    return s.bit_count() if hasattr(int,'bit_count') else bin(s).count('1')

def read_pred(x):
    if isinstance(x,dict): out=x
    elif isinstance(x,list):
        out={}
        for r in x:
            if not isinstance(r,dict) or 'id' not in r or 'summary' not in r: raise ValueError('Each prediction item must contain id and summary.')
            if r['id'] in out: raise ValueError(f'Duplicate prediction ID: {r["id"]}')
            out[r['id']]=r['summary']
    else: raise ValueError('Prediction must be a JSON object or list of objects.')
    for k,v in out.items():
        if not isinstance(k,str) or not isinstance(v,str) or not v.strip(): raise ValueError(f'Invalid summary for ID {k!r}.')
    return out

def main():
    ap=argparse.ArgumentParser(description='Evaluate Phase 6 summaries on your own held-out split.')
    ap.add_argument('--truth',required=True); ap.add_argument('--prediction',required=True); a=ap.parse_args()
    t=load(a.truth); p=read_pred(load(a.prediction))
    if not isinstance(t,dict) or not t: raise ValueError('Truth must be a non-empty JSON object.')
    if set(t)!=set(p): raise ValueError(f"ID mismatch: missing={len(set(t)-set(p))}, extra={len(set(p)-set(t))}")
    preds=[p[k].strip() for k in t]; refs=[t[k].strip() for k in t]
    scorer=rouge_scorer.RougeScorer(['rouge1'],use_stemmer=True); tok=tokenizers.DefaultTokenizer(use_stemmer=True)
    r1=rl=0.0
    for pr,rf in zip(preds,refs):
        s=scorer.score(rf,pr); r1+=s['rouge1'].fmeasure
        pt=tok.tokenize(pr); rt=tok.tokenize(rf)
        if not pt and not rt: x=1.0
        elif not pt or not rt: x=0.0
        else:
            z=lcs(pt,rt); pp=z/len(pt); rr=z/len(rt); x=0.0 if pp+rr==0 else 2*pp*rr/(pp+rr)
        rl+=x
    n=len(t); r1/=n; rl/=n
    try:
        import torch; device='cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError: device=None
    _,_,f=bert_score.score(preds,refs,lang='en',batch_size=8,verbose=False,**({'device':device} if device else {}))
    print(f'Documents: {n}'); print(f'BERTScore F1: {float(f.mean().item()):.6f}'); print(f'ROUGE-1: {r1:.6f}'); print(f'ROUGE-L: {rl:.6f}')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'EVALUATION FAILED: {e}',file=sys.stderr); sys.exit(1)
