# Phase 4: Court-judgment classification

The task predicts one of two judgment classes for an entire legal document. Documents are split into context-sized chunks, each chunk is encoded, and chunk representations are combined before classification. The experiments compare backbones and adaptation methods, then test context aggregation: mean, attention, weighted and maximum pooling; overlap; selected document sections; a hierarchical GRU; and different classification heads. Calibration is treated as a separate output decision.

The best recorded submission, **1008827**, uses the rebuilt dense decoder with LoRMA+, mean chunk pooling and a linear classifier. It scored **0.6879 Macro-F1**. The source contains the context, head and training implementations, but the per-run configuration files and checkpoints are not bundled.

**Implementation:** [data processing](source/src/phase4_data.py), [document classifier](source/src/phase4_model.py), [context strategies](source/src/phase4_context.py), [alternative head](source/src/phase4_dense_mlp.py), [training](source/src/phase4_training.py), [experiment logic](source/src/phase4_deadline.py).

## Recorded Codabench submissions

These statuses and scores come from the reviewed submission ledger. A dash means no finished score was recorded at that audit. Variants in the table are short descriptions; source modules show the implementations.

| ID | Backbone | Method | Variant | Last audited status | Primary score |
|---:|---|---|---|---|---:|
| 1007756 | GQA decoder | partial layerwise fine-tuning | mean chunk pooling, linear head | Finished | 0.4533 |
| 1007788 | MLA decoder | LoRMA+ | mean chunk pooling, linear head | Finished | 0.4885 |
| 1007805 | MHA decoder | full fine-tuning | mean chunk pooling, linear head | Finished | 0.3887 |
| 1007928 | rebuilt dense decoder | LoRMA+ | mean chunk pooling, linear head | Finished | 0.5618 |
| 1007935 | MQA decoder | knowledge distillation | mean chunk pooling, linear head | Finished | 0.3766 |
| 1008001 | encoder-decoder | partial layerwise fine-tuning | mean chunk pooling, linear head | Finished | 0.4587 |
| 1008188 | rebuilt dense decoder | LoRMA+ | MLP head | Finished | 0.3323 |
| 1008471 | MLA decoder | LoRMA+ | mean chunk pooling, linear head | Finished | 0.5713 |
| 1008550 | rebuilt dense decoder | LoRMA+ | calibration | Finished | 0.5929 |
| 1008553 | rebuilt dense decoder | LoRMA+ | weighted-pool inference | Finished | 0.5596 |
| 1008554 | rebuilt dense decoder | LoRMA+ | max-pool inference | Finished | 0.5410 |
| 1008822 | rebuilt dense decoder | LoRMA+ | attention pooling | Finished | 0.6575 |
| 1008824 | rebuilt dense decoder | LoRMA+ | 800-token context | Finished | 0.6261 |
| 1008827 | rebuilt dense decoder | LoRMA+ | mean chunk pooling, linear head | Finished | 0.6879 |
| 1008828 | rebuilt dense decoder | LoRMA+ | ending-only context | Finished | 0.6119 |
| 1008832 | rebuilt dense decoder | LoRMA+ | hierarchical GRU pooling | Finished | 0.6313 |
| 1008834 | rebuilt dense decoder | LoRMA+ | 256-token overlap | Finished | 0.6587 |
| 1008835 | rebuilt dense decoder | LoRMA+ | residual MLP head | Finished | 0.6616 |
| 1008838 | rebuilt dense decoder | LoRMA+ | selected document sections | Finished | 0.5591 |

## Reproduction boundary

The code here is a source snapshot. Exact historical runs require their original data, run-specific settings, tokenizer/checkpoint pairing and saved weights. Those inputs are not all distributed in this repository. The linked files document the implementation; this README does not claim that the scored submissions can be regenerated from source alone.
