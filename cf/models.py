"""Reuse the local ItemGCN models without copying or changing their losses."""
import hashlib
import importlib
import sys
import types
from pathlib import Path

from .data import file_hash

PAIRED_FAMILIES = ('lightgcn-rns', 'stabcf', 'simgcl', 'directau')
MODEL_FAMILIES = PAIRED_FAMILIES + ('fism',)


def model_variants(family, phase):
    if phase == 'native':
        return ['original'] if family == 'fism' else []
    return ['original', 'item-only'] if phase == 'both' else [phase]


def source_package(root):
    root = Path(root).resolve()
    if not (root / 'modules' / 'LightGCN.py').is_file():
        raise FileNotFoundError('Missing ItemGCN model source: ' + str(root))
    # Isolated namespace: do not shadow AlphaFree's models or another repo's modules.
    name = '_alphafree_itemgcn_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root / 'modules')]
        sys.modules[name] = package
    return name


def source_hashes(root):
    return {p.name: file_hash(p) for p in sorted((Path(root) / 'modules').glob('*.py'))}


def build_model(data, args, model_name, variant, device, graph=None, aggregation=None):
    if model_name == 'fism':
        if variant not in ('original', 'native', 'item-only'):
            raise ValueError('Unknown FISM variant: ' + variant)
        from .fism import FISM, FISMB
        return (FISMB(data, args) if variant == 'item-only' else FISM(data, args)).to(device)
    module = importlib.import_module(source_package(args.itemgcn_root) + '.LightGCN')
    classes = {'lightgcn-rns': ('LightGCN', 'XLightGCN'),
               'stabcf': ('LightGCN', 'XLightGCN'),
               'simgcl': ('SimGCL', 'XSimGCL'), 'directau': ('DirectAU', 'XDirectAU')}
    item_only = variant == 'item-only'
    cls = getattr(module, classes[model_name][int(item_only)])
    config = types.SimpleNamespace(**vars(args))
    # Legacy StabCF constructors hardcode cuda:0. Construct on CPU, register graph
    # state as buffers, then move all state together to the requested device.
    config.cuda = False
    config.ns = 'rns' if model_name == 'lightgcn-rns' else 'stabcf'
    graph = data.graph() if graph is None else graph
    inputs = [dict(n_users=data.n_users, n_items=data.n_items), config, graph]
    if item_only:
        inputs.append(data.aggregation(args.b_mode, args.b_a, args.b_b)
                      if aggregation is None else aggregation)
    model = cls(*inputs)
    if model_name in ('lightgcn-rns', 'stabcf'):
        for name in ('sparse_norm_adj', 'si_norm_mat'):
            if hasattr(model, name):
                value = getattr(model, name)
                delattr(model, name)
                model.register_buffer(name, value.coalesce(), persistent=False)
        model.gcn.interact_mat = model.sparse_norm_adj
    model = model.to(device)
    has_user_table = any(name == 'user_embed' for name, _ in model.named_parameters())
    if has_user_table == item_only:
        raise AssertionError('Unexpected learnable user table for ' + variant)
    return model
