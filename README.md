# CS781 Legal NLP Capstone

Rishit Dutta · 240870

This repository contains the model and task code for the completed capstone experiments. Each phase folder has a README that explains its task, methods, source files and recorded Codabench submissions. The submission tables distinguish completed scores from submissions that were still pending or failed at the last audit.

| Phase | Task | Best scored student submission | Recorded primary score |
|---|---|---:|---:|
| [1](phase1/README.md) | Legal language-model pretraining | 962550 (MLA) | 0.7947 |
| [2](phase2/README.md) | Named-entity recognition | 977556 (dense, LoRMA+, boundary head) | 0.4097 |
| [3](phase3/README.md) | Rhetorical-role prediction | 991958 (GQA, LoRA) | 0.4028 |
| [4](phase4/README.md) | Court-judgment classification | 1008827 (dense, LoRMA+) | 0.6879 |
| [5](phase5/README.md) | Statute identification | 1014156 (GQA, full fine-tuning) | 0.3177 |
| [6](phase6/README.md) | Abstractive summarization | 1028478 (dense, full fine-tuning) | 0.8292 |

The Phase 1 results table includes the official starter as an attributed baseline; the table above selects the best scored student model. Phase 7 preparation did not produce a completed submission and is not represented as a finished phase here. Scores are the recorded Codabench primary metrics, which differ by task and should not be compared across phases.

## How the phases connect

Phase 1 trains legal language models and byte-BPE tokenizers. Later phases initialize from the relevant Phase 1 checkpoint and add a task-specific head or adaptation: token and boundary prediction in Phase 2, sentence classification in Phase 3, document classification in Phase 4, multilabel classification in Phase 5, and generation in Phase 6. The phase READMEs identify the backbone and adaptation used by each submission.

## Repository layout

- `phaseN/source/src/`: model, data, training and prediction modules for that phase.
- `phaseN/source/scripts/`: preparation or command-line scripts where included.
- `phaseN/source/tokenizers/`: tokenizer JSON files used by the corresponding code.
- `phaseN/source/official/evaluation_scripts/`: supplied evaluation scripts where included.
- `requirements.txt`: Python package dependencies. Install a compatible PyTorch build separately, then install these packages in an isolated environment.

The source tree repeats some foundation modules across phase folders so each phase can be read in context. The [official gated dataset](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) requires authorized access. Raw data, trained checkpoints, per-run configuration files and generated prediction archives are not included in this repository. The tokenizer JSON files are included. Consequently, this source snapshot documents the implementations but does not by itself reproduce an exact scored submission. The phase READMEs point to the relevant implementation files and do not present missing launch commands as runnable.

## Reading the results

Each phase README lists every recorded submission ID, its architecture, adaptation or context variant, the last audited status, and a score only when the submission was marked finished. The full experiment analysis and limitations are in the submitted capstone report. Official starter and evaluation code remain attributed to the course; the byte-BPE and Transformer foundations build on Assignment 1 and Assignment 3 work.
