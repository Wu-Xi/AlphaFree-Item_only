"""Test only the original and validation-selected variant from a completed Alpha search."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from cf.data import CFData
from cf.evaluate import evaluate
from models.AlphaFree3 import AlphaFree3, load_frozen_state
from models.base.user_aggregation import build_user_aggregation
from search_alphafree2 import ROOT, exclusive_lock, file_hash, read_json, summary, write_json


def select_runs(root):
    manifest_path = root / 'manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('version') != 2:
        raise ValueError('Expected original + 28 search manifest version 2')
    # Fail closed if data, training code, or inference code changed since the search.
    for relative, expected in manifest['source_sha256'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Source changed since training: ' + relative)
    for path, expected in manifest['data_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('Dataset/cache changed since training: ' + path)
    for dataset, expected in manifest['config_sha256'].items():
        if file_hash(ROOT / 'configs' / (dataset + '.yaml')) != expected:
            raise ValueError('Dataset configuration changed since training: ' + dataset)
    report = summary(root, manifest['jobs'])
    if not report['complete']:
        raise ValueError('Finish all original + 28 jobs before selecting final test models')
    selected = []
    for group in report['groups']:
        for role, row in [('original', group['original']), ('selected_variant', group['best_variant'])]:
            if row is None:
                raise ValueError('Missing original or variant for ' + group['dataset'])
            job = next(j for j in manifest['jobs'] if j['id'] == row['id'])
            status = read_json(root / (job['id'] + '.json'))
            checkpoint = row['result']['checkpoint']
            actual = file_hash(checkpoint)
            if status.get('checkpoint_sha256') != actual:
                raise ValueError('Checkpoint changed or missing fingerprint: ' + checkpoint)
            selected.append(dict(role=role, job=job, result=row['result'], checkpoint_sha256=actual))
    return dict(search_manifest_sha256=file_hash(manifest_path), selected=selected,
                selection='Original plus best of 28 variants by validation Recall@20, per dataset')


def build_frozen_model(row, data, device, encode_batch_size):
    """Reuse the frozen MLP evaluator, explicitly install the TRAINING aggregation."""
    args = row['job']['parameters']
    dataset = Path(args['data_path']) / args['dataset']
    filename = 'item_cf_embeds_large3_array.npy' if args['lm_model'] == 'v3' else 'item_cf_embeds_LLAMA_array.npy'
    features = np.load(dataset / 'item_info' / filename, allow_pickle=False)
    if features.ndim != 2 or features.shape[0] != data.n_items:
        raise ValueError('Feature/catalog mismatch')
    model = AlphaFree3(features, load_frozen_state(row['result']['checkpoint']),
                      data.train, data.n_users, device, encode_batch_size)
    if args['model_name'] == 'AlphaFree':
        model.aggregation = sparse.load_npz(dataset / 'preprocessed/user_interact_matrix.npz').tocsr()
    else:
        model.aggregation = build_user_aggregation(data.train, data.n_users, data.n_items,
            args['b_mode'], args['b_a'], args['b_b'], preserve_counts=True)
    if model.aggregation.shape != (data.n_users, data.n_items):
        raise ValueError('Aggregation/catalog mismatch')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--encode-batch-size', type=int, default=1024)
    parser.add_argument('--execute', action='store_true', help='Without this flag, only verify and preview selection')
    args = parser.parse_args(argv)
    if min(args.threads, args.eval_batch_size, args.encode_batch_size) < 1:
        parser.error('Batch sizes and threads must be positive')
    root = args.search_root.resolve()
    selection = select_runs(root)
    if not args.execute:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return selection
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('A usable CPU/CUDA device is required')
    torch.set_num_threads(args.threads)
    output = (args.output or root / 'final_test').resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / 'running.lock'):
        frozen = dict(selection=selection, device=str(device), torch=str(torch.__version__),
                      eval_batch_size=args.eval_batch_size, encode_batch_size=args.encode_batch_size,
                      threads=args.threads, split='test', protocol='unique-item truth; AlphaFree C++ masks')
        path = output / 'selection.json'
        if path.exists() and read_json(path) != frozen:
            raise ValueError('Final test selection/runtime changed; use a NEW output directory')
        # Freeze BEFORE reading any test metrics.
        if not path.exists():
            write_json(path, frozen)
        results = []
        for row in selection['selected']:
            job, params = row['job'], row['job']['parameters']
            result_path = output / (job['id'] + '.json')
            if result_path.exists():
                result = read_json(result_path)
                if result['selected'] != row:
                    raise ValueError('Saved test result does not match locked selection')
            else:
                data = CFData(Path(params['data_path']) / params['dataset'] / 'cf_data', duplicate_items='count')
                model = build_frozen_model(row, data, device, args.encode_batch_size)
                metrics = evaluate(model, data, split='test', k=20, batch_size=args.eval_batch_size)
                result = dict(selected=row, metrics=metrics, split='test')
                write_json(result_path, result)
                del model, data
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            results.append(result)
            print(job['id'] + ': ' + json.dumps(result['metrics']), flush=True)
        report = dict(selection=selection, results=results, split='test', complete=True)
        write_json(output / 'summary.json', report)
        return report


if __name__ == '__main__':
    main()
