# Phase 1: Legal language-model pretraining

The task is to continue legal text from a prompt. The student experiments train decoder-only Transformers with MHA, GQA, MQA and MLA attention. A separate official GPT-2 starter provides a reference score and is identified as course code. The student models use an Assignment-1-style 16k byte-BPE tokenizer; tokenized training text is packed into fixed-length sequences for next-token prediction. A later rebuilt model and a full-corpus encoder-decoder model were used as starting points in later phases but were not additional scored Phase 1 submissions.

The first four student submissions each processed about 200M training-token targets from a reused 100M-token shard.

**Implementation:** [tokenizer training](source/scripts/train_capstone_bpe.py), [shard construction](source/scripts/build_phase1_shards.py), [decoder training](source/scripts/train_phase1_model.py), [prediction generation](source/scripts/generate_phase1_predictions.py), [rebuilt models](source/src/phase1_v2_models.py), [encoder-decoder model](source/src/phase1_full_encdec.py).

The submission JSON maps each official test ID to a continuation. The course evaluator measures BERTScore-F1. The tokenizer JSON is included, while the trained checkpoints and original token shards are external inputs.

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 954177 | GPT-2 starter | pretraining | Official starter (course baseline) | Finished | 0.8008 |
| 962547 | GQA decoder | pretraining | next-token prediction | Finished | 0.7875 |
| 962549 | MHA decoder | pretraining | next-token prediction | Finished | 0.7841 |
| 962550 | MLA decoder | pretraining | next-token prediction | Finished | 0.7947 |
| 962552 | MQA decoder | pretraining | next-token prediction | Finished | 0.7876 |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
