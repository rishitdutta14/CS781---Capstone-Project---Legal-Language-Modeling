"""Independent GQA adaptation experiments, initialized ONLY from Phase1 weights."""

from contextlib import contextmanager
import datetime
from pathlib import Path
import time

import torch
from torch import nn

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase2_adaptation import LoRMAPlusLinear
from src.phase3_model import load_backbone
from src.phase5_model import StatuteClassifier
from src.phase5_tasks import StatuteData, package_submission, predictions
import src.phase5_checks as checks
import src.phase5_training as training


class AdapterFFN(nn.Module):
    """One residual bottleneck on each FFN output; zero up-projection is identity.

    This is a declared single-adapter-per-FFN variant, not an exact reproduction
    of the original two-adapter Houlsby placement. No additional dropout or
    normalization is introduced, and the original FFN remains frozen.
    """
    def __init__(self, original, width, bottleneck=64):
        super().__init__()
        if not 0 < bottleneck <= width:
            raise ValueError('Invalid adapter bottleneck width')
        self.original = original.requires_grad_(False)
        self.down = nn.Linear(width, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.original(x)
        return y + self.up(self.activation(self.down(y)))


class GQAMethodClassifier(StatuteClassifier):
    def __init__(self, backbone, method, precision='bf16', rank=8, alpha=8.,
                 bottleneck=64, head_dropout=.1):
        if backbone.cfg.attention_type != 'standard' or backbone.cfg.n_kv_heads != 2:
            raise ValueError('This campaign keeps the original two-KV-head GQA architecture')
        # Construct the same fresh 100-label head before configuring adaptation.
        super().__init__(backbone, 'gqa', 'full_fine_tuning', precision=precision, head_dropout=head_dropout)
        self.method = method
        if method == 'lorma_plus':
            backbone.requires_grad_(False)
            for index, block in enumerate(backbone.blocks):
                for name in ('q_proj', 'v_proj'):
                    setattr(block.attn, name, LoRMAPlusLinear(getattr(block.attn, name), rank, alpha))
                    self.targets.append(f'backbone.blocks.{index}.attn.{name}')
        elif method == 'bottleneck_adapters':
            backbone.requires_grad_(False)
            for index, block in enumerate(backbone.blocks):
                block.ffn = AdapterFFN(block.ffn, backbone.cfg.d_model, bottleneck)
                self.targets.append(f'backbone.blocks.{index}.ffn')
        elif method != 'full_fine_tuning':
            raise ValueError('Only full fine-tuning, LoRMA+, and bottleneck adapters are in scope')


def build(config, device):
    if config.get('initialization') != 'own_Phase1_GQA_only_fresh_task_head':
        raise ValueError('Never substitute the Phase5 epoch26 checkpoint for Phase1 initialization')
    if file_sha256(Path(config['checkpoint'])) != config['checkpoint_sha256']:
        raise ValueError('Frozen Phase1 GQA checkpoint changed')
    backbone = load_backbone(config['checkpoint'], config['model_config'])
    return GQAMethodClassifier(backbone, config['method'], config['compute_precision'], config['rank'],
                               config['alpha'], config['adapter_bottleneck'], config['head_dropout']).to(device)


def export_milestone(output, model, progress, config):
    """Export a committed validation-selected model; never inspect test labels."""
    data = StatuteData(config['data_manifest'])
    folder = Path(output) / 'milestones' / f'best_epoch{progress["best_epoch"]:03d}_{time.time_ns()}'
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(Path(output) / 'progress.json', {**progress, 'stage': 'exporting_new_validation_best',
                                               'milestone_folder': str(folder)})
    logits = training.infer(model, data, 'test', config, folder / 'progress.json')
    records = predictions(data.rows['test'], training.sigmoid(logits), progress['best_threshold'])
    atomic_json(folder / 'prediction.json', records)
    archive = package_submission(folder / 'submission.zip', records, [row['id'] for row in data.rows['test']])
    training.write_array(folder / 'test_logits.npy', logits)
    checkpoint = Path(output) / 'best.pt'
    result = {'name': config['name'], 'method': config['method'], 'best_epoch': progress['best_epoch'],
              'best_dev_macro_f1': progress['best_score'], 'threshold': progress['best_threshold'],
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
              'checkpoint_reference_note': 'SHA at export time; only running best/latest weights are retained, not every milestone model',
              'submission': archive, 'all_cases_all_chunks': True, 'uploaded': False,
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(folder / 'completed.json', result)
    atomic_json(Path(output) / 'latest_ready_submission.json', result)
    return result


@contextmanager
def method_functions(config, export_ready=False):
    """Process-local entry points, leaving every frozen older source unchanged."""
    old_build, old_check_build, old_save = training.build, checks.build, training.save
    def save_with_export(output, model, optimizer, progress, digest, improved):
        old_save(output, model, optimizer, progress, digest, improved)
        if export_ready and improved and progress.get('epoch', 0) > 0:
            mode = model.training
            try:
                export_milestone(output, model, progress, config)
            finally:
                model.train(mode)
    training.build = build
    checks.build = build
    training.save = save_with_export
    try:
        yield
    finally:
        training.build, checks.build, training.save = old_build, old_check_build, old_save


def pilot(config_path, device):
    config = training.read_json(config_path)
    with method_functions(config):
        result = checks.pilot(config_path, device)
    result['initialization'] = 'Own Phase1 GQA, fresh100-label head; no Phase5 checkpoint loaded'
    result['new_original_backbone_parameters_are_frozen_for_PEFT'] = config['method'] != 'full_fine_tuning'
    atomic_json(Path(config['output']) / 'pilot/completed.json', result)
    return result


def train(config_path, device, resume=False):
    config = training.read_json(config_path)
    with method_functions(config, export_ready=True):
        return training.train(config_path, device, resume)
