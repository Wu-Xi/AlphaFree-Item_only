"""Frozen AlphaFree evaluation: single B, 29-B validation search, or selected-B test."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from cf.data import CFData, DATASETS, file_hash
from cf.evaluate import evaluate
from models.AlphaFree3 import AlphaFree3, load_frozen_state
from models.base.user_aggregation import aggregation_grid

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='Movie')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--checkpoint', type=Path, help='Default: data-root/pretrained/<dataset>.pth.tar')
    p.add_argument('--features', type=Path, help='Override original language-feature npy file')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--encode-batch-size', type=int, default=1024)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--b-mode', choices=['sum', 'mean', 'sqrt', 'sym', 'degree', 'weighted_mean'], default='mean')
    p.add_argument('--b-a', type=float, default=.5)
    p.add_argument('--b-b', type=float, default=.5)
    p.add_argument('--search', action='store_true', help='Evaluate all 29 B settings on validation ONLY')
    p.add_argument('--split', choices=['valid', 'test'], default='valid')
    p.add_argument('--selection', type=Path, help='Completed validation search selection.json; requires --split test')
    p.add_argument('--eval-users', type=int, default=0, help='SMOKE ONLY: first N evaluation users')
    p.add_argument('--max-settings', type=int, default=0, help='Limit newly evaluated settings; 0 means unlimited')
    p.add_argument('--output', type=Path, help='Separate output directory; matching completed settings may be resumed')
    args = p.parse_args(argv)
    args.dataset = DATASETS.get(args.dataset.lower(), args.dataset)
    if args.dataset not in DATASETS.values():
        p.error('Unknown dataset')
    if min(args.threads, args.eval_batch_size, args.encode_batch_size, args.top_k) < 1 or min(args.eval_users, args.max_settings) < 0:
        p.error('Invalid batch sizes, threads, top-k or smoke/run limits')
    if not all(math.isfinite(v) and v >= 0 for v in (args.b_a, args.b_b)):
        p.error('B exponents must be finite and nonnegative')
    if args.search and (args.split != 'valid' or args.selection):
        p.error('29-B search is validation-only and cannot use --selection')
    if args.selection and args.split != 'test':
        p.error('--selection requires --split test')
    args.data_root = args.data_root.resolve()
    args.checkpoint = (args.checkpoint or args.data_root / 'pretrained' / (args.dataset + '.pth.tar')).resolve()
    name = ('item_cf_embeds_large3_array.npy' if args.dataset in ('amazon_movie', 'amazon_book_2014')
            else 'item_cf_embeds_LLAMA_array.npy')
    args.features = (args.features or args.data_root / args.dataset / 'item_info' / name).resolve()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    args.output = (args.output or ROOT / 'weights/alphafree3' / args.dataset / stamp).resolve()
    return args


def choose_best(results, k):
    # Deterministic ties: search evaluates mean first, then the existing grid order.
    return max(results, key=lambda row: row['metrics']['recall@{}'.format(k)])


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('A usable CPU/CUDA device is required')
    data = CFData(args.data_root / args.dataset / 'cf_data')
    fingerprint = dict(dataset=args.dataset, checkpoint_sha256=file_hash(args.checkpoint),
                       features_sha256=file_hash(args.features), data_sha256=data.stats['sha256'])
    sources = [Path(__file__), ROOT / 'models/AlphaFree3.py', ROOT / 'models/base/user_aggregation.py',
               ROOT / 'cf/data.py', ROOT / 'cf/evaluate.py']
    source_hashes = {str(path.resolve().relative_to(ROOT)): file_hash(path) for path in sources}
    selection = read_json(args.selection) if args.selection else None
    if selection is not None:
        if (selection.get('model') != 'AlphaFree3' or not selection.get('complete')
                or selection.get('smoke_test') is not False or selection.get('split') != 'valid'
                or selection.get('planned') != 29 or selection.get('top_k') != args.top_k
                or selection.get('fingerprint') != fingerprint
                or selection.get('source_sha256') != source_hashes):
            raise ValueError('Selection must be a complete, non-smoke, matching 29-B validation search')
        settings = [selection['setting']]
        if settings[0] not in aggregation_grid():
            raise ValueError('Selected B is not in the 29-setting grid')
    elif args.search:
        settings = sorted(aggregation_grid(), key=lambda setting: setting['id'] != 'mean')
    else:
        settings = [dict(id='single', mode=args.b_mode, a=args.b_a, b=args.b_b)]
    manifest = dict(model='AlphaFree3', version=1, fingerprint=fingerprint, source_sha256=source_hashes,
                    settings=settings, split=args.split, search=args.search, top_k=args.top_k,
                    eval_users=args.eval_users, eval_batch_size=args.eval_batch_size,
                    encode_batch_size=args.encode_batch_size, threads=args.threads, device=str(device),
                    python=sys.version.split()[0], torch=str(torch.__version__),
                    selection_sha256=file_hash(args.selection) if args.selection else None)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / 'running.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        path = args.output / 'manifest.json'
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError('Manifest changed: use a NEW output directory')
        else:
            write_json(path, manifest)
        print('AlphaFree3 FROZEN EVALUATION; no training or parameter updates.', flush=True)
        if args.eval_users:
            print('SMOKE TEST: truncated users; results cannot select a final B.', flush=True)
        print('Checkpoint: ' + str(args.checkpoint), flush=True)
        print('Output: ' + str(args.output), flush=True)
        results, model, parameter_hash, attempts = [], None, None, 0
        for setting in settings:
            result_path = args.output / (setting['id'] + '.json')
            if result_path.exists():
                result = read_json(result_path)
                if (result.get('setting') != setting or result.get('split') != args.split
                        or result.get('smoke_test') != bool(args.eval_users)
                        or result.get('fingerprint') != fingerprint
                        or not math.isfinite(result['metrics']['recall@{}'.format(args.top_k)])):
                    raise ValueError('Invalid saved result: ' + str(result_path))
                results.append(result)
                print('SKIP ' + setting['id'], flush=True)
                continue
            if args.max_settings and attempts >= args.max_settings:
                continue
            if model is None:
                features = np.load(args.features, allow_pickle=False)
                if features.ndim != 2 or features.shape[0] != data.n_items:
                    raise ValueError('Language-feature rows do not match the item catalog')
                state = load_frozen_state(args.checkpoint)
                model = AlphaFree3(features, state, data.train, data.n_users, device, args.encode_batch_size)
                del features, state
                parameter_hash = model.parameter_hash()
                print('Trainable parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)), flush=True)
            model.set_aggregation(setting['mode'], setting['a'], setting['b'])
            metrics = evaluate(model, data, args.split, args.top_k, args.eval_batch_size, args.eval_users)
            result = dict(model='AlphaFree3', setting=setting, split=args.split, metrics=metrics,
                          smoke_test=bool(args.eval_users), fingerprint=fingerprint,
                          trainable_parameters=0, parameter_sha256=parameter_hash)
            write_json(result_path, result)
            results.append(result)
            attempts += 1
            print(setting['id'] + ': ' + json.dumps(metrics), flush=True)
        if model is not None and (model.parameter_hash() != parameter_hash or any(p.requires_grad for p in model.parameters())):
            raise AssertionError('Frozen model parameters changed during evaluation')
        complete = len(results) == len(settings)
        best = choose_best(results, args.top_k) if args.search and results and not args.eval_users else None
        report = dict(model='AlphaFree3', mode='frozen-eval', split=args.split,
                      planned=len(settings), completed=len(results), complete=complete,
                      smoke_test=bool(args.eval_users), best_validation=best,
                      selection='validation Recall only; incomplete best is provisional; smoke has no best',
                      results=results)
        write_json(args.output / 'summary.json', report)
        if args.search and complete and not args.eval_users:
            chosen = dict(model='AlphaFree3', complete=True, smoke_test=False, planned=29,
                          split='valid', top_k=args.top_k, setting=best['setting'],
                          validation_metrics=best['metrics'], fingerprint=fingerprint, source_sha256=source_hashes)
            write_json(args.output / 'selection.json', chosen)
            print('Selected on validation: ' + best['setting']['id'], flush=True)
        return report
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
