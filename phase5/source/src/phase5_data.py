"""Raw, text-preserving LSI input audit. Test non-input fields are never used."""

from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from src.phase1_full_corpus_prepare import atomic_json,file_sha256,iter_json_records

REPO='Exploration-Lab/CS-781-Capstone'
REVISION='93b0059be756c6c461f3a225de25aabda7d5f84a'
FILES={'train':'Phase_5_LSI/Train/train.json','dev':'Phase_5_LSI/Train/dev.json','test':'Phase_5_LSI/Test/test.json'}


def project_record(record,split,index):
    if split not in FILES:raise ValueError('Unknown split')
    if not isinstance(record,dict):raise ValueError('Each example must be an object')
    identifier=record['id'];text=record['text']
    if not isinstance(identifier,str) or not identifier.strip() or not isinstance(text,str):
        raise ValueError('Invalid case ID or text')
    projected={'id':identifier,'row_key':f'{split}:{index}','text':text}
    if split!='test':
        labels=record['labels']
        if not isinstance(labels,list) or any(type(v) is not int or v<0 for v in labels):
            raise ValueError('Labels must be nonnegative integer statute IDs')
        # Repeated IDs in official input are an audit finding, not grounds for
        # rewriting targets here. Output predictions will still require uniqueness.
        projected['labels']=list(labels)
    return projected


def quantiles(values):
    return {'min':int(min(values)),'p50':float(np.quantile(values,.5)),
        'p95':float(np.quantile(values,.95)),'max':int(max(values)),'sum':int(sum(values))}


def audit(files,output):
    """Retain every official row; create whitelisted JSONL views and evidence."""
    output=Path(output);output.mkdir(exist_ok=False)
    ids=defaultdict(list);texts=defaultdict(list);summaries={};views={};counts={};pairs=Counter()
    for split in ('train','dev','test'):
        source=Path(files[split]['path']);view=output/f'{split}.jsonl'
        characters=[];byte_lengths=[];cardinality=[];raw_cardinality=[];label_counts=Counter();empty=0;nonascii=0;index=[];repeated_labels=[]
        with view.open('x',encoding='utf-8') as stream:
            for number,raw in enumerate(iter_json_records(source)):
                row=project_record(raw,split,number);text=row['text'];digest=hashlib.sha256(text.encode()).hexdigest()
                entry={'split':split,'row_key':row['row_key'],'id':row['id'],'text_sha256':digest}
                if split!='test':
                    entry['labels']=list(row['labels']);unique=set(row['labels']);label_counts.update(unique)
                    cardinality.append(len(unique));raw_cardinality.append(len(row['labels']))
                    if len(unique)!=len(row['labels']):repeated_labels.append({'row_key':row['row_key'],'id':row['id'],'raw_labels':row['labels']})
                    if split=='train':pairs.update(itertools.combinations(sorted(unique),2))
                ids[row['id']].append(entry);texts[digest].append(entry)
                characters.append(len(text));byte_lengths.append(len(text.encode()));empty+=not bool(text.strip())
                nonascii+=not text.isascii()
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                index.append(entry)
                if number%1000==0:atomic_json(output/'progress.json',{'split':split,'rows_done':number+1,
                    'test_noninput_fields_used':False,'GPU_training_started':False})
        if not characters:raise ValueError('Official split is empty')
        atomic_json(output/f'{split}_index.json',index)
        counts[split]=label_counts
        summaries[split]={'rows':len(index),'unique_IDs':len({r['id'] for r in index}),
            'repeated_ID_rows_retained':len(index)-len({r['id'] for r in index}),
            'empty_text_rows_retained':int(empty),'rows_containing_non_ASCII':int(nonascii),
            'characters':quantiles(characters),'UTF8_bytes':quantiles(byte_lengths),
            'label_fields_examined':split!='test','labels':None if split=='test' else {
                'observed_IDs':sorted(label_counts),'frequencies':dict(sorted(label_counts.items())),
                'label_cardinality':quantiles(cardinality),'mean_labels_per_document':sum(cardinality)/len(cardinality),
                'frequency_semantics':'Positive documents per unique statute ID; repeated IDs within a raw row count once in statistics only',
                'raw_list_cardinality':quantiles(raw_cardinality),'rows_with_repeated_label_IDs':repeated_labels,
                'repeated_label_occurrences':sum(raw_cardinality)-sum(cardinality),
                'empty_label_set_rows':sum(v==0 for v in cardinality),
                'multi_label_rows':sum(v>1 for v in cardinality)}}
        views[split]={'path':str(view),'sha256':file_sha256(view),'bytes':view.stat().st_size,
            'index':str(output/f'{split}_index.json'),'index_sha256':file_sha256(output/f'{split}_index.json')}
    duplicate_ids=[v for v in ids.values() if len(v)>1]
    cross_ids=[v for v in duplicate_ids if len({r['split'] for r in v})>1]
    cross_texts=[v for v in texts.values() if len({r['split'] for r in v})>1]
    conflicts=[]
    for group in duplicate_ids:
        labeled=[r for r in group if r['split']!='test']
        if len({r['text_sha256'] for r in group})>1 or len({tuple(sorted(set(r['labels']))) for r in labeled})>1:
            conflicts.append(group)
    overlap={'cross_split_ID_groups':cross_ids,'cross_split_exact_text_groups':cross_texts,
        'conflicting_repeated_ID_groups':conflicts,
        'test_repeated_ID_groups':[v for v in duplicate_ids if sum(r['split']=='test' for r in v)>1]}
    atomic_json(output/'overlaps.json',overlap)
    train_ids=set(counts['train']);dev_ids=set(counts['dev']);all_ids=train_ids|dev_ids
    dimension=max(all_ids)+1 if all_ids else 0
    label_audit={'observed_training_IDs':sorted(train_ids),'observed_development_IDs':sorted(dev_ids),
        'development_only_IDs':sorted(dev_ids-train_ids),'training_only_IDs':sorted(train_ids-dev_ids),
        'max_observed_label_ID_plus_one':dimension,
        'gaps_in_zero_to_max_observed_range':sorted(set(range(dimension))-all_ids),
        'train_labels_with_at_most_5_positive_rows':sorted(k for k,n in counts['train'].items() if n<=5),
        'train_labels_with_at_most_20_positive_rows':sorted(k for k,n in counts['train'].items() if n<=20),
        'top_train_cooccurring_label_pairs':[{'IDs':list(k),'rows':n} for k,n in pairs.most_common(20)],
        'official_evaluator_dimension_caveat':'Local evaluator uses max reference gold ID+1; this is not independent proof of the global valid-ID vocabulary',
        'train_reference_dimension':max(train_ids)+1 if train_ids else 0,
        'dev_reference_dimension':max(dev_ids)+1 if dev_ids else 0,
        'test_label_fields_examined':False}
    result={'splits':summaries,'label_audit':label_audit,'overlap_counts':{k:len(v) for k,v in overlap.items()},
        'inputs':views,'raw_text_cleaned_or_normalized':False,'rows_dropped_or_deduplicated':False,
        'raw_label_lists_preserved_in_views':True,
        'train_dev_resplit':False,'model_inputs':['text'],'test_labels_or_other_metadata_used':False,
        'test_safe_view_fields':['id','row_key','text'],
        'tokenization_done':False,'token_lengths_measured':False,'GPU_training_started':False,
        'training_readiness':'Raw audit only; resolve duplicate/overlap/label findings before model training'}
    atomic_json(output/'data_audit.json',result)
    return result
