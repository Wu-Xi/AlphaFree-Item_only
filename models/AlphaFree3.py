"""Frozen AlphaFree checkpoint with inference-only user aggregation replacement.

No optimizer, training forward, augmentation data or language-model dependencies.
Keep both MLPs solely to load the original checkpoint strictly; inference uses
only mlp_origin. Semantic features and normalized item outputs are fixed buffers.
"""
import hashlib
import inspect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base.user_aggregation import build_user_aggregation


def load_frozen_state(path):
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    checkpoint = torch.load(path, **options)
    state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected an AlphaFree state_dict or checkpoint containing state_dict')
    return state


class AlphaFree3(nn.Module):
    def __init__(self, features, state, train_history, n_users, device='cpu', encode_batch_size=1024):
        super().__init__()
        if encode_batch_size < 1:
            raise ValueError('encode_batch_size must be positive')
        required = 'mlp_origin.0.weight'
        if required not in state or state[required].ndim != 2:
            raise ValueError('Checkpoint is missing the AlphaFree original-view MLP')
        hidden, input_dim = state[required].shape
        if 'mlp_origin.2.weight' not in state or state['mlp_origin.2.weight'].ndim != 2:
            raise ValueError('Checkpoint is missing the AlphaFree output layer')
        output_dim = state['mlp_origin.2.weight'].shape[0]
        if features.ndim != 2 or features.shape[1] != input_dim:
            raise ValueError('Language-feature dimension does not match the checkpoint')
        self.n_users, self.n_items = n_users, features.shape[0]
        self.train_history = train_history
        for name in ('mlp_origin', 'mlp_aug'):
            setattr(self, name, nn.Sequential(nn.Linear(input_dim, hidden), nn.LeakyReLU(),
                                             nn.Linear(hidden, output_dim)))
        self.load_state_dict(state, strict=True)
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Checkpoint contains nonfinite parameters')
        self.requires_grad_(False)
        self.register_buffer('features', torch.as_tensor(np.asarray(features, dtype=np.float32)), persistent=False)
        if not torch.isfinite(self.features).all():
            raise ValueError('Language features contain nonfinite values')
        self.to(device)
        self.eval()
        with torch.no_grad():
            items = torch.cat([F.normalize(self.mlp_origin(self.features[start:start + encode_batch_size]), dim=-1)
                               for start in range(0, self.n_items, encode_batch_size)])
        self.register_buffer('item_vectors', items, persistent=False)
        self.set_aggregation('mean')

    def train(self, mode=True):
        if mode:
            raise RuntimeError('AlphaFree3 is frozen inference only; use AlphaFree2 for retraining')
        return super().train(False)

    def forward(self, *args, **kwargs):
        raise RuntimeError('AlphaFree3 has no training loss; use score_all or user_vectors')

    def set_aggregation(self, mode='mean', a=.5, b=.5):
        self.aggregation = build_user_aggregation(self.train_history, self.n_users, self.n_items, mode, a, b)
        self.setting = dict(mode=mode, a=a, b=b)

    @torch.no_grad()
    def user_vectors(self, users):
        if isinstance(users, torch.Tensor):
            users = users.detach().cpu().numpy()
        matrix = self.aggregation[users].tocoo()
        indices = torch.as_tensor(np.stack([matrix.row, matrix.col]), dtype=torch.long,
                                  device=self.features.device)
        values = torch.as_tensor(matrix.data, dtype=self.features.dtype, device=self.features.device)
        sparse_b = torch.sparse_coo_tensor(indices, values, matrix.shape, device=self.features.device)
        profile = torch.sparse.mm(sparse_b, self.features)
        return F.normalize(self.mlp_origin(profile), dim=-1)

    @torch.no_grad()
    def score_all(self, users):
        return self.user_vectors(users) @ self.item_vectors.T

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
