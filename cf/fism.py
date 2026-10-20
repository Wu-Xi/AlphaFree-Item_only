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
    aggregation_protocol = 'exclude all target occurrences; count-weighted history; residual event count ^ (-alpha)'

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
        self.register_buffer('history_counts', torch.tensor(csr.data, dtype=torch.long), persistent=False)
        self.register_buffer('user_counts', torch.tensor(csr.sum(axis=1).A1, dtype=torch.long), persistent=False)
        rows = torch.repeat_interleave(torch.arange(data.n_users), torch.tensor(csr.indptr[1:] - csr.indptr[:-1]))
        self.register_buffer('edge_keys', rows * data.n_items + self.history_items, persistent=False)
        self.src = nn.Embedding(data.n_items, args.dim)
        self.dst = nn.Embedding(data.n_items, args.dim)
        nn.init.normal_(self.src.weight, std=.01)
        nn.init.normal_(self.dst.weight, std=.01)
        self.item_bias = nn.Parameter(torch.zeros(data.n_items))
        self.user_bias = nn.Parameter(torch.zeros(data.n_users)) if args.fism_user_bias else None

    def history_indices(self, users):
        starts = self.history_offsets[users]
        counts = self.history_offsets[users + 1] - starts
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        rows = torch.repeat_interleave(torch.arange(len(users), device=users.device), counts)
        positions = starts[rows] + torch.arange(rows.numel(), device=users.device) - offsets[rows]
        items = self.history_items[positions]
        return self.user_counts[users], offsets, rows, items, self.history_counts[positions]

    def history(self, users):
        counts, offsets, rows, items, repeats = self.history_indices(users)
        sums = F.embedding_bag(items, self.src.weight, offsets, mode='sum', include_last_offset=True,
                               per_sample_weights=repeats.to(self.src.weight.dtype))
        return sums, counts, rows, items, repeats

    def contains(self, users, items):
        return self.multiplicity(users, items) > 0

    def multiplicity(self, users, items):
        keys = users * self.n_items + items
        if self.edge_keys.numel() == 0:
            return torch.zeros_like(keys)
        indices = torch.searchsorted(self.edge_keys, keys.contiguous())
        bounded = indices.clamp_max(self.edge_keys.numel() - 1)
        present = (indices < self.edge_keys.numel()) & (self.edge_keys[bounded] == keys)
        return self.history_counts[bounded] * present

    def score_targets(self, users, targets):
        """Score a [batch, candidates] matrix with candidate-specific exclusion."""
        sums, counts, _, _, _ = self.history(users)
        repeats = self.multiplicity(users[:, None], targets)
        profiles = sums[:, None, :] - self.src(targets) * repeats[:, :, None]
        lengths = (counts[:, None] - repeats).clamp_min(1).to(sums.dtype)
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
        sums, counts, rows, items, repeats = self.history(users)
        scores = (sums @ self.dst.weight.T) * counts.clamp_min(1).to(sums.dtype).pow(-self.alpha)[:, None]
        # Exact self-exclusion also for overlapping train/eval targets.
        corrected = ((sums[rows] - self.src(items) * repeats[:, None]) * self.dst(items)).sum(-1)
        corrected = corrected * (counts[rows] - repeats).clamp_min(1).to(sums.dtype).pow(-self.alpha)
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


class FISMB(FISM):
    """FISM-BCE with candidate-excluded B aggregation, not a new user-table removal.

    Item degrees always use the full training graph. User normalization uses
    the remaining history after excluding the candidate, so sqrt recovers
    original FISM alpha=.5. weighted_mean renormalizes remaining item weights.
    fism_alpha is deliberately NOT applied on top of B.
    """
    aggregation_protocol = 'exclude all target occurrences; count-weighted history and train degrees; residual user degree/weight mass'

    def __init__(self, data, args):
        super().__init__(data, args)
        self.mode = args.b_mode
        presets = dict(sum=(0., 0.), mean=(1., 0.), sqrt=(.5, 0.), sym=(.5, .5))
        if self.mode in presets:
            self.a, self.b = presets[self.mode]
        elif self.mode == 'degree':
            self.a, self.b = args.b_a, args.b_b
        elif self.mode == 'weighted_mean':
            self.a, self.b = 0., args.b_b
        else:
            raise ValueError('Unknown FISM B mode')
        degrees = torch.tensor(data.interactions.sum(axis=0).A1, dtype=torch.float32)
        weights = torch.zeros_like(degrees)
        nonempty = degrees > 0
        weights[nonempty] = degrees[nonempty].pow(-self.b)
        self.register_buffer('item_weights', weights, persistent=False)

    def weighted_history(self, users):
        counts, offsets, rows, items, repeats = self.history_indices(users)
        weights = self.item_weights[items] * repeats
        sums = F.embedding_bag(items, self.src.weight, offsets, mode='sum',
                               per_sample_weights=weights, include_last_offset=True)
        mass = weights.new_zeros(len(users)).index_add(0, rows, weights)
        return sums, counts, mass, rows, items, repeats

    def scale(self, counts, mass):
        if self.mode == 'weighted_mean':
            # A positive weighted mass can be <1: clamp_min(1) would be incorrect.
            return torch.where(mass > 0, mass.clamp_min(torch.finfo(mass.dtype).tiny).reciprocal(),
                               torch.zeros_like(mass))
        return counts.clamp_min(1).to(mass.dtype).pow(-self.a)

    def score_targets(self, users, targets):
        sums, counts, mass, _, _, _ = self.weighted_history(users)
        repeats = self.multiplicity(users[:, None], targets)
        removed_weight = self.item_weights[targets] * repeats
        profiles = sums[:, None, :] - self.src(targets) * removed_weight[:, :, None]
        scale = self.scale(counts[:, None] - repeats, mass[:, None] - removed_weight)
        scores = (profiles * self.dst(targets)).sum(-1) * scale + self.item_bias[targets]
        if self.user_bias is not None:
            scores = scores + self.user_bias[users, None]
        return scores

    def score_all(self, users):
        users = torch.as_tensor(users, dtype=torch.long, device=self.src.weight.device)
        sums, counts, mass, rows, items, repeats = self.weighted_history(users)
        scores = (sums @ self.dst.weight.T) * self.scale(counts, mass)[:, None]
        removed_weight = self.item_weights[items] * repeats
        corrected = ((sums[rows] - self.src(items) * removed_weight[:, None])
                     * self.dst(items)).sum(-1)
        corrected = corrected * self.scale(counts[rows] - repeats, mass[rows] - removed_weight)
        scores[rows, items] = corrected
        scores = scores + self.item_bias[None, :]
        if self.user_bias is not None:
            scores = scores + self.user_bias[users, None]
        return scores
