"""User-list input, train-only sparse graphs, and ItemGCN-compatible sampling."""
import hashlib
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


DATASETS = dict(movie='amazon_movie', book='amazon_book_2014',
                video='amazon_video', baby='amazon_baby', steam='steam',
                beauty='amazon_beauty_personal', health='amazon_health')


def read_split(path, duplicate_items='error'):
    if duplicate_items not in ('error', 'count'):
        raise ValueError('duplicate_items must be error or count')
    histories = {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = [int(x) for x in line.split()]
            if min(values) < 0:
                raise ValueError('{}:{}: negative ID'.format(path, line_number))
            user, items = values[0], values[1:]
            if user in histories or (duplicate_items == 'error' and len(items) != len(set(items))):
                raise ValueError('{}:{}: duplicate user row or item'.format(path, line_number))
            histories[user] = items
    return histories


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CFData:
    def __init__(self, directory, duplicate_items='error'):
        self.directory = Path(directory).resolve()
        self.splits = {name: read_split(self.directory / (name + '.txt'), duplicate_items)
                       for name in ('train', 'valid', 'test')}
        all_users = set().union(*(set(split) for split in self.splits.values()))
        all_items = {item for split in self.splits.values()
                     for items in split.values() for item in items}
        if not all_users or not all_items:
            raise ValueError('Empty user/item universe')
        # IDs retain their original values; held-out splits only define the ID universe.
        self.n_users, self.n_items = max(all_users) + 1, max(all_items) + 1
        if len(all_users) != self.n_users or len(all_items) != self.n_items:
            raise ValueError('Expected contiguous zero-based user/item IDs across the three splits')
        self.train = self.splits['train']
        self.train_sets = {u: set(items) for u, items in self.train.items()}
        self.pairs = np.asarray([(u, i) for u, items in self.train.items()
                                 for i in items], dtype=np.int64).reshape(-1, 2)
        if not len(self.pairs):
            raise ValueError('Training split has no interactions')
        self.interactions = sp.csr_matrix(
            (np.ones(len(self.pairs), dtype=np.float32),
             (self.pairs[:, 0], self.pairs[:, 1])), shape=(self.n_users, self.n_items))
        self.stats = dict(n_users=self.n_users, n_items=self.n_items,
                          duplicate_items=duplicate_items,
                          repeated_occurrences={s: sum(len(items) - len(set(items)) for items in h.values())
                                                for s, h in self.splits.items()},
                          observed_user_ids=len(all_users), observed_item_ids=len(all_items),
                          interactions={s: sum(map(len, h.values())) for s, h in self.splits.items()},
                          sha256={s: file_hash(self.directory / (s + '.txt')) for s in self.splits})

    def graph(self):
        normalized = self.aggregation('sym')
        return sp.bmat([[None, normalized], [normalized.T, None]], format='coo', dtype=np.float32)

    def aggregation(self, mode='sym', a=0.5, b=0.5):
        presets = dict(sum=(0., 0.), mean=(1., 0.), sqrt=(0.5, 0.), sym=(0.5, 0.5))
        if mode in presets:
            a, b = presets[mode]
        elif mode == 'weighted_mean':
            a = 0.
        elif mode != 'degree':
            raise ValueError('Unknown aggregation: ' + mode)
        if not np.isfinite([a, b]).all() or min(a, b) < 0:
            raise ValueError('Degree exponents must be finite and nonnegative')
        r = self.interactions
        du, di = np.asarray(r.sum(1)).ravel(), np.asarray(r.sum(0)).ravel()
        wu, wi = np.zeros(self.n_users), np.zeros(self.n_items)
        wu[du > 0], wi[di > 0] = du[du > 0] ** -a, di[di > 0] ** -b
        result = sp.diags(wu).dot(r).dot(sp.diags(wi))
        if mode == 'weighted_mean':
            rowsum = np.asarray(result.sum(1)).ravel()
            inv = np.zeros(self.n_users)
            inv[rowsum > 0] = 1. / rowsum[rowsum > 0]
            result = sp.diags(inv).dot(result)
        return result.astype(np.float32).tocoo()

    def batch(self, pairs, model, device, window_length):
        batch = dict(users=torch.as_tensor(pairs[:, 0], device=device),
                     pos_items=torch.as_tensor(pairs[:, 1], device=device))
        if not model.requires_negative_sampling:
            return batch
        count = model.negative_sample_count
        negatives = np.random.randint(self.n_items, size=(len(pairs), count), dtype=np.int64)
        for row, user in enumerate(pairs[:, 0]):
            positives = self.train_sets[user]
            if len(positives) >= self.n_items:
                raise ValueError('User {} has no negative item'.format(user))
            for col in range(count):
                while negatives[row, col] in positives:
                    negatives[row, col] = np.random.randint(self.n_items)
        batch['neg_items'] = torch.as_tensor(negatives, device=device)
        if getattr(model, 'requires_history', False):
            # Same boundary and replacement rule as ItemGCN/main.py.
            observed = [random.sample(self.train[u], k=window_length)
                        if len(self.train[u]) > window_length
                        else random.choices(self.train[u], k=window_length) for u in pairs[:, 0]]
            batch['observed_pos_items'] = torch.as_tensor(
                np.asarray(observed, dtype=np.int64), device=device)
        return batch
