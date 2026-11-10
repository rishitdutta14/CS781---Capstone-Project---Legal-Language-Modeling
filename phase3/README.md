# Phase 3: Rhetorical-role prediction

The task assigns one rhetorical-role label to each sentence in a legal document. Sentence text is encoded with its surrounding document context. The model pools a representation for the target sentence and applies a classification head; long sentences are covered in pieces rather than silently truncated. All 13 role labels remain in Macro-F1, including classes absent from a development split.

The selected submission, **991958**, adapts the original GQA decoder with LoRA and scored **0.4028 Macro-F1**. The other submissions compare MQA with distillation, MLA with IA3, MHA with LoRMA+, rebuilt dense and encoder-decoder models with LoRA, and the later full-corpus encoder-decoder model.

**Implementation:** [data handling](source/src/phase3_data.py), [model](source/src/phase3_model.py), [training](source/src/phase3_training.py), [rebuilt model](source/src/phase3_rebuilt_model.py), [rebuilt training](source/src/phase3_rebuilt_training.py), [data setup](source/scripts/setup_phase3_data.py), [token-length audit](source/scripts/audit_phase3_token_lengths.py), [submission validation](source/scripts/validate_phase3_submission.py).

This source snapshot does not include the original run configurations or trained weights; it should not be treated as a standalone command to recreate the scored ZIPs.

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 991958 | GQA decoder | LoRA | sentence classification | Finished | 0.4028 |
| 991960 | MQA decoder | knowledge distillation | sentence classification | Finished | 0.3958 |
| 991961 | MLA decoder | IA3 | sentence classification | Finished | 0.3821 |
| 991963 | MHA decoder | LoRMA+ | sentence classification | Finished | 0.3630 |
| 992589 | rebuilt dense decoder | LoRA | sentence classification | Finished | 0.3955 |
| 992592 | encoder-decoder | LoRA | sentence classification | Finished | 0.3809 |
| 992598 | full-corpus encoder-decoder | LoRA | sentence classification | Finished | 0.0818 |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
