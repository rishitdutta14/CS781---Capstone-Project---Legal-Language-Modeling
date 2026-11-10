"""The original GQA full-tuning recipe with training-only positive BCE weights."""

from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
import src.phase5_checks as checks
import src.phase5_gqa_methods as gqa
import src.phase5_model as model_module
import src.phase5_training as training
from src.phase5_prepare import binary_target

FORMULA = 'min(10, max(1, sqrt((N - positive_documents) / positive_documents)))'


def weight_record(rows):
    if not rows:
        raise ValueError('Cannot fit class weights on an empty training split')
    counts = Counter()
    for row in rows:
        target = binary_target(row['labels'])
        counts.update(target.nonzero()[0].tolist())
    size = len(rows)
    values = [min(10., max(1., math.sqrt((size - counts[c]) / counts[c])))
              if counts[c] else 10. for c in range(100)]
    return {'partition': 'train', 'documents': size, 'formula': FORMULA,
            'positive_documents': [counts[c] for c in range(100)], 'pos_weight': values,
            'negative_weight': 1., 'zero_positive_labels': [c for c in range(100) if not counts[c]],
            'repeated_annotations_count_once': True,
            'zero_positive_policy': 'Use the cap10; there are no positive targets for that label'}


def load_weighting(config):
    if (config['architecture'] != 'gqa' or config['method'] != 'full_fine_tuning'
            or config['initialization'] != 'own_Phase1_GQA_only_fresh_task_head'
            or config['loss'] != 'train_frequency_weighted_BCE' or config['kd_weight'] != 0.):
        raise ValueError('This experiment changes only the supervised BCE objective')
    path = Path(config['positive_weights_path'])
    if file_sha256(path) != config['positive_weights_sha256']:
        raise ValueError('Frozen positive weights changed')
    record = json.loads(path.read_text())
    manifest = json.loads(Path(config['data_manifest']).read_text())
    train = manifest['splits']['train']
    index = Path(train['index'])
    if (file_sha256(index) != train['index_sha256']
            or record['train_index_sha256'] != train['index_sha256']):
        raise ValueError('Weights no longer refer to the frozen training split')
    expected = weight_record(json.loads(index.read_text()))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Weights do not match unique positive TRAIN-document frequencies')
    return record


def make_weighted_loss(values):
    weights = torch.as_tensor(values, dtype=torch.float32).detach().cpu().clone()
    if weights.shape != (100,) or not torch.isfinite(weights).all() or ((weights < 1) | (weights > 10)).any():
        raise ValueError('Expected100 finite positive weights in [1,10]')
    devices = {}

    def loss(logits, labels, teacher=None, temperature=2., kd_weight=.5):
        if logits.shape != (100,) or teacher is not None:
            raise ValueError('Weighted GQA uses100 independent supervised logits and no teacher')
        key = str(logits.device)
        if key not in devices:
            devices[key] = weights.to(logits.device)
        target = torch.as_tensor(binary_target(labels), device=logits.device)
        supervised = F.binary_cross_entropy_with_logits(
            logits.float(), target, pos_weight=devices[key], reduction='mean')
        return supervised, supervised.detach(), supervised.detach().new_zeros(())

    return loss


@contextmanager
def weighted_objective(config):
    """Use the same weighted loss in joint/replayed training and every pilot gate."""
    record = load_weighting(config)
    original_model, original_checks = model_module.document_loss, checks.document_loss
    weighted = make_weighted_loss(record['pos_weight'])
    model_module.document_loss = checks.document_loss = weighted
    try:
        yield record
    finally:
        model_module.document_loss, checks.document_loss = original_model, original_checks


def pilot(config_path, device):
    config = json.loads(Path(config_path).read_text())
    with weighted_objective(config) as weights:
        result = gqa.pilot(config_path, device)
    result.update(loss=config['loss'], positive_weights_sha256=config['positive_weights_sha256'],
                  weight_fitting_partition=weights['partition'])
    atomic_json(Path(config['output']) / 'pilot/completed.json', result)
    return result


def train(config_path, device, resume=False):
    config = json.loads(Path(config_path).read_text())
    with weighted_objective(config), gqa.method_functions(
            config, export_ready=config.get('export_each_validation_improvement', False)):
        result = training.train(config_path, device, resume)
    result.update(loss=config['loss'], positive_weights_sha256=config['positive_weights_sha256'],
                  training_loss_note='Training BCE includes positive weights; development BCE remains unweighted',
                  selection='Original development Macro-F1 and global threshold grid',
                  export_each_validation_improvement=config.get('export_each_validation_improvement', False))
    atomic_json(Path(config['output']) / 'production/completed.json', result)
    return result


def recover(config_path, device):
    """Check the resumed next update twice, then roll back both diagnostic steps."""
    config = json.loads(Path(config_path).read_text())
    training.verify_bindings(config)
    output = Path(config['output']) / 'production'
    digest = file_sha256(Path(config_path))
    with weighted_objective(config), gqa.method_functions(config, export_ready=False):
        model = gqa.build(config, device)
        optimizer = training.make_optimizer(model, config)
        progress = training.load(output / 'latest.pt', model, optimizer, digest)
        data = training.StatuteData(config['data_manifest'])
        result = checks.recovery_gate(output, model, optimizer, digest,
                                      [data.chunks('train', 0)[0][:64]], data.rows['train'][0]['labels'])
    result.update(progress_preserved=progress, config_sha256=digest,
                  diagnostic_updates_rolled_back=True, export_each_validation_improvement=False)
    atomic_json(output / 'retention_recovery.json', result)
    if not result['passed']:
        raise ValueError('Retention-resume exact optimizer/RNG recovery failed')
    return result
