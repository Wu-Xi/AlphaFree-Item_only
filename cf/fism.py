"""Native FISM-BCE, matching the local MixGCF FISM objective and target exclusion.

Two item tables; optional scalar user biases, never a user embedding table.
Histories contain training interactions only and are reconstructed on reload.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FISM(nn.Module):
    requires_negative_sampling = True
    requires_history = False

    def __init__(self, data, args):
        super().__init__()
        self.n_items = data.n_items
        self.alpha = args.fism_alpha
        self.reg_weight = args.fism_reg
        self.negative_sample_count = args.fism_negatives
        csr = data.interactions.copy()
        csr.sort_indices()
        self.register_buffer('history_items', torch.tensor(csr.indices, dtype=torch.long), persistent=False)
        self.register_buffer('history_offsets', torch.tensor(csr.indptr, dtype=torch.long), persistent=False)
        rows = torch.repeat_interleave(torch.arange(data.n_users), torch.tensor(csr.indptr[1:] - csr.indptr[:-1]))
        self.register_buffer('edge_keys', rows * data.n_items + self.history_items, persistent=False)
        self.src = nn.Embedding(data.n_items, args.dim)
        self.dst = nn.Embedding(data.n_items, args.dim)
        nn.init.normal_(self.src.weight, std=.01)
        nn.init.normal_(self.dst.weight, std=.01)
        self.item_bias = nn.Parameter(torch.zeros(data.n_items))
        self.user_bias = nn.Parameter(torch.zeros(data.n_users)) if args.fism_user_bias else None

    def history(self, users):
        starts = self.history_offsets[users]
        counts = self.history_offsets[users + 1] - starts
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        rows = torch.repeat_interleave(torch.arange(len(users), device=users.device), counts)
        positions = starts[rows] + torch.arange(rows.numel(), device=users.device) - offsets[rows]
        items = self.history_items[positions]
        sums = F.embedding_bag(items, self.src.weight, offsets, mode='sum', include_last_offset=True)
        return sums, counts, rows, items

    def contains(self, users, items):
        keys = users * self.n_items + items
        if self.edge_keys.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        indices = torch.searchsorted(self.edge_keys, keys.contiguous())
        return ((indices < self.edge_keys.numel()) &
                (self.edge_keys[indices.clamp_max(self.edge_keys.numel() - 1)] == keys))

    def score_targets(self, users, targets):
        """Score a [batch, candidates] matrix with candidate-specific exclusion."""
        sums, counts, _, _ = self.history(users)
        present = self.contains(users[:, None], targets)
        profiles = sums[:, None, :] - self.src(targets) * present[:, :, None]
        lengths = (counts[:, None] - present.long()).clamp_min(1).to(sums.dtype)
        scores = (profiles * self.dst(targets)).sum(-1) * lengths.pow(-self.alpha)
        scores = scores + self.item_bias[targets]
        if self.user_bias is not None:
            scores = scores + self.user_bias[users, None]
        return scores

    def score_pairs(self, users, items):
        return self.score_targets(users, items[:, None]).squeeze(1)

    def score_all(self, users):
        """Full-catalog scores without a batch x catalog x dimension tensor."""
        users = torch.as_tensor(users, dtype=torch.long, device=self.src.weight.device)
        sums, counts, rows, items = self.history(users)
        scores = (sums @ self.dst.weight.T) * counts.clamp_min(1).to(sums.dtype).pow(-self.alpha)[:, None]
        # Exact self-exclusion also for overlapping train/eval targets.
        corrected = ((sums[rows] - self.src(items)) * self.dst(items)).sum(-1)
        corrected = corrected * (counts[rows] - 1).clamp_min(1).to(sums.dtype).pow(-self.alpha)
        scores[rows, items] = corrected
        scores = scores + self.item_bias[None, :]
        if self.user_bias is not None:
            scores = scores + self.user_bias[users, None]
        return scores

    def forward(self, batch, epoch=0):
        targets = torch.cat([batch['pos_items'][:, None], batch['neg_items']], dim=1)
        logits = self.score_targets(batch['users'], targets)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.
        recommendation = F.binary_cross_entropy_with_logits(logits, labels)
        # Deliberately unsquared global norms, as in local MixGCF/modules/FISM.py.
        regularization = self.reg_weight * (self.src.weight.norm(2) + self.dst.weight.norm(2))
        return recommendation + regularization, recommendation, regularization
