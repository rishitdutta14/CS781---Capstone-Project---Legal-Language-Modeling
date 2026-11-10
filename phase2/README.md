# Phase 2: Legal named-entity recognition

The task identifies entity types and exact character spans in legal documents. The pipeline tokenizes each document, uses overlapping windows for long documents, predicts BIO tags, and converts token predictions back to original character coordinates. The stronger dense-model variants add a learned boundary prediction so entity starts and ends can fall inside BPE tokens. Validation determines checkpoint selection and decoding choices.

The selected submission, **977556**, uses the rebuilt dense Phase 1 decoder with LoRMA+ adaptation, BIO classification and a boundary head. Its recorded strict Macro-F1 is **0.4097**. Other entries compare MHA/GQA/MQA/MLA, an encoder-decoder model, plain BIO, and knowledge distillation.

**Implementation:** [data preparation](source/scripts/prepare_phase2_data.py), [BIO training](source/scripts/train_phase2_bio.py), [boundary training](source/scripts/train_phase2_boundary.py), [offset mapping](source/src/tokenization/a1_offsets.py), [boundary logic](source/src/phase2_boundaries.py), [adaptations](source/src/phase2_adaptation.py), [decoding](source/src/phase2_decoding.py).

The supplied Phase 2 evaluator is under `source/official/evaluation_scripts/`. Exact reproduction requires the official data, the corresponding Phase 1 weights and the run-specific settings, which are not bundled here.

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 975933 | rebuilt dense decoder | full fine-tuning | Plain BIO baseline | Finished | 0.2050 |
| 976126 | rebuilt dense decoder | full fine-tuning | BIO with boundary prediction | Finished | 0.3580 |
| 976221 | MQA decoder | full fine-tuning | plain BIO | Finished | 0.1684 |
| 977396 | MHA decoder | full fine-tuning | plain BIO | Finished | 0.1604 |
| 977401 | GQA decoder | full fine-tuning | plain BIO | Finished | 0.1807 |
| 977402 | MLA decoder | full fine-tuning | plain BIO | Finished | 0.1728 |
| 977556 | rebuilt dense decoder | LoRMA+ | BIO with boundary prediction | Finished | 0.4097 |
| 977985 | encoder-decoder | partial layerwise fine-tuning | BIO with boundary prediction | Finished | 0.1492 |
| 978328 | MQA decoder | knowledge distillation | BIO with boundary prediction | Failed | — |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
