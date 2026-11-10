"""Two-linear-layer head ablation, without editing live campaign sources."""

import contextlib

from torch import nn

import src.phase4_deadline as deadline
from src.phase3_rebuilt_model import load_rebuilt_backbone


class DenseMLPClassifier(deadline.DeadlineClassifier):
    def __init__(self,backbone,precision='bf16',dropout=.1):
        # Construct the same baseline first, preserving its adapter
        # initialization sequence. Only then replace its linear head.
        super().__init__(backbone,'dense','lorma_plus',precision)
        width=backbone.cfg.d_model
        self.classifier=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(256,2))
        self.head_type='mlp'


def build(config,device):
    if config['architecture']!='dense' or config['method']!='lorma_plus' or config['head']!='mlp':
        raise ValueError('Only the requested dense decoder LoRMA MLP ablation is allowed')
    backbone=load_rebuilt_backbone(config['checkpoint'],config['model_config'],config['checkpoint_sha256'])
    return DenseMLPClassifier(backbone,config['compute_precision'],config['head_dropout']).to(device)


@contextlib.contextmanager
def builder():
    original=deadline.build
    deadline.build=build
    try:yield
    finally:deadline.build=original


def pilot(path,device):
    with builder():return deadline.pilot(path,device)


def train(path,device):
    with builder():return deadline.train(path,device)
