"""Train-only user aggregation, using the same 29 settings as the CF experiments."""
import itertools

import numpy as np
from scipy import sparse


def build_user_aggregation(train_history, n_users, n_items, mode='mean', a=.5, b=.5,
                           preserve_counts=False):
    if not np.isfinite([a, b]).all() or min(a, b) < 0:
        raise ValueError('B exponents must be finite and nonnegative')
    presets = {'sum': (0., 0.), 'mean': (1., 0.), 'sqrt': (.5, 0.), 'sym': (.5, .5)}
    if mode in presets:
        a, b = presets[mode]
    elif mode == 'weighted_mean':
        a = 0.
    elif mode != 'degree':
        raise ValueError('Unknown B mode: ' + str(mode))
    rows, cols = [], []
    for user, items in train_history.items():
        if not 0 <= user < n_users:
            raise ValueError('User ID outside catalog')
        # AlphaFree2 uses counts, matching original AlphaFree. Keep the legacy
        # default for the independent AlphaFree3 experiment.
        items = sorted(items) if preserve_counts else sorted(set(items))
        if any(not 0 <= item < n_items for item in items):
            raise ValueError('Item ID outside catalog')
        rows.extend([user] * len(items))
        cols.extend(items)
    r = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n_users, n_items))
    du, di = np.asarray(r.sum(1)).ravel(), np.asarray(r.sum(0)).ravel()
    wu, wi = np.zeros(n_users), np.zeros(n_items)
    wu[du > 0], wi[di > 0] = du[du > 0] ** -a, di[di > 0] ** -b
    result = sparse.diags(wu).dot(r).dot(sparse.diags(wi))
    if mode == 'weighted_mean':
        rowsum = np.asarray(result.sum(1)).ravel()
        inv = np.zeros(n_users)
        inv[rowsum > 0] = 1. / rowsum[rowsum > 0]
        result = sparse.diags(inv).dot(result)
    return result.astype(np.float32).tocsr()


def aggregation_grid():
    values = [0., .25, .5, .75, 1.]
    presets = {(0., 0.): 'sum', (1., 0.): 'mean', (.5, 0.): 'sqrt', (.5, .5): 'sym'}
    result = []
    for a, b in itertools.product(values, repeat=2):
        mode = presets.get((a, b), 'degree')
        label = mode if mode != 'degree' else 'degree_a{:g}_b{:g}'.format(a, b).replace('.', 'p')
        result.append(dict(id=label, mode=mode, a=a, b=b))
    for b in values[1:]:
        result.append(dict(id='weighted_mean_b{:g}'.format(b).replace('.', 'p'),
                           mode='weighted_mean', a=0., b=b))
    return result
