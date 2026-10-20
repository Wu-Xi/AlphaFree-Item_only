"""Full-catalog evaluation matching AlphaFree's masks, Recall and NDCG."""
import numpy as np
import torch


def excluded_items(data, split, user):
    other = 'test' if split == 'valid' else 'valid'
    truth = set(data.splits[split].get(user, []))
    return (set(data.train.get(user, [])) | set(data.splits[other].get(user, []))) - truth


def rank_metrics(rank, truth, k):
    truth = set(truth)
    if not truth:
        raise ValueError('Metrics require nonempty ground truth')
    hits = np.array([item in truth for item in rank[:k]], dtype=np.float64)
    recall = hits.sum() / len(truth)
    dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
    idcg = (1. / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
    return np.array([recall, dcg / idcg, float(hits.sum() > 0)])


@torch.no_grad()
def evaluate(model, data, split='valid', k=20, batch_size=128, max_users=0):
    if split not in ('valid', 'test'):
        raise ValueError('Evaluation split must be valid or test')
    model.eval()
    native_scores = hasattr(model, 'score_all')
    if not native_scores:
        users_e, items_e = model.generate()
        if not torch.isfinite(users_e).all() or not torch.isfinite(items_e).all():
            raise FloatingPointError('Nonfinite evaluation embeddings')
    users = sorted(u for u, items in data.splits[split].items() if items)
    if max_users:
        users = users[:max_users]
    if not users:
        raise ValueError('No users with ground truth in ' + split)
    total = np.zeros(3)
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        scores = (model.score_all(batch_users) if native_scores else
                  model.rating(users_e[batch_users], items_e))
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite ranking scores')
        for row, user in enumerate(batch_users):
            excluded = list(excluded_items(data, split, user))
            if excluded:
                scores[row, excluded] = -torch.inf
        values, ranks = torch.topk(scores, min(k, data.n_items), dim=1)
        values, ranks = values.cpu().numpy(), ranks.cpu().numpy()
        for row, user in enumerate(batch_users):
            rank = ranks[row][np.isfinite(values[row])]
            total += rank_metrics(rank, data.splits[split][user], k)
    total /= len(users)
    return {'recall@{}'.format(k): float(total[0]), 'ndcg@{}'.format(k): float(total[1]),
            'hr@{}'.format(k): float(total[2]), 'users': len(users)}
