"""AlphaFree with configurable train-only B in the ORIGINAL user branch.

U = mlp_origin(B @ X). Augmented users/items, losses and prediction are inherited
unchanged. Default mean recovers AlphaFree with matching train-only caches.
"""
from .AlphaFree import AlphaFree
from .base.user_aggregation import build_user_aggregation


class AlphaFree2(AlphaFree):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.user_interact_matrix = build_user_aggregation(
            data.train_user_list, data.n_users, data.n_items,
            args.b_mode, args.b_a, args.b_b)
