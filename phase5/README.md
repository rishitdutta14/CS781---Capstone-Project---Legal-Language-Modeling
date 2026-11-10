# Phase 5: Legal statute identification

The task predicts any applicable statutes from a 100-label set for each legal document. Each document is covered in chunks. Chunk representations are pooled, and a multilabel head predicts one probability per statute. The experiments compare pretrained backbones, full and partial fine-tuning, LoRMA+, bottleneck adapters and distillation. Later GQA experiments also compare class-weighted BCE, asymmetric loss and attention pooling. Prediction thresholds are selected on development data, not test labels.

The best finished submission in the reviewed records, **1014156**, uses the original Phase 1 GQA backbone with full fine-tuning and a new multilabel head. It selected epoch 13 and a global threshold of 0.15, scoring **0.3177 Macro-F1**. Some submitted entries were not marked finished in the reviewed status snapshot; their rows below have no finished score.

**Implementation:** [data](source/src/phase5_data.py), [model](source/src/phase5_model.py), [training](source/src/phase5_training.py), [GQA adaptation methods](source/src/phase5_gqa_methods.py), [weighted-loss experiment](source/src/phase5_weighted_gqa.py), [GQA ablations](source/src/phase5_gqa_ablations.py), [preparation](source/src/phase5_prepare.py).

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 1014149 | MHA decoder | full fine-tuning | mean chunk pooling, multilabel head | Preparing | — |
| 1014150 | GQA decoder | partial layerwise fine-tuning | mean chunk pooling, multilabel head | Submitted | — |
| 1014151 | MLA decoder | LoRMA+ | mean chunk pooling, multilabel head | Failed | — |
| 1014152 | MQA decoder | knowledge distillation | mean chunk pooling, multilabel head | Finished | 0.3083 |
| 1014153 | rebuilt dense decoder | LoRMA+ | mean chunk pooling, multilabel head | Finished | 0.3126 |
| 1014154 | encoder-decoder | partial layerwise fine-tuning | mean chunk pooling, multilabel head | Finished | 0.3025 |
| 1014156 | GQA decoder | full fine-tuning | mean chunk pooling, multilabel head | Finished | 0.3177 |
| 1014161 | GQA decoder | LoRMA+ | mean chunk pooling, multilabel head | Submitting | — |
| 1014162 | GQA decoder | bottleneck adapters | mean chunk pooling, multilabel head | Finished | 0.3027 |
| 1016607 | GQA decoder | full fine-tuning | class-weighted BCE | Finished | 0.3068 |
| 1016813 | GQA decoder | full fine-tuning | asymmetric loss, mean pooling | Finished | 0.3092 |
| 1016815 | GQA decoder | full fine-tuning | BCE, attention pooling | Finished | 0.2993 |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
