# Phase 6: Abstractive legal summarization

The task generates a concise summary from a legal judgment. Long source documents require context selection. The dense-decoder approach uses opening, middle and closing token regions; training predicts summary tokens while masking source-token loss. Other submissions compare MHA with LoRA, GQA with adapters, MLA with LoRMA, MQA with distillation, and a full-corpus encoder-decoder model.

The best recorded submission, **1028478**, uses the rebuilt dense decoder with full fine-tuning, selected at epoch 10. It scored **0.8292 BERTScore-F1**. Its source context selects 384 opening, 256 middle and 384 closing tokens. The training data preparation removes empty-reference rows and splits remaining cases by document hash.

**Implementation:** [source and target preparation](source/src/phase6_data.py), [generation model](source/src/phase6_model.py), [training and decoding](source/src/phase6_training.py), [encoder-decoder foundation](source/src/phase1_full_encdec.py).

The official dataset and original trained weights are needed to regenerate predictions; they are not in this repository.

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 1028478 | rebuilt dense decoder | full fine-tuning | context-selected summarization | Finished | 0.8292 |
| 1028479 | MHA decoder | LoRA | summarization | Finished | 0.7968 |
| 1028481 | GQA decoder | adapter | summarization | Finished | 0.8013 |
| 1028482 | MLA decoder | LoRMA | summarization | Finished | 0.7416 |
| 1028483 | MQA decoder | knowledge distillation | summarization | Finished | 0.8191 |
| 1028484 | encdec251 | full fine-tuning | summarization | Finished | 0.7345 |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
