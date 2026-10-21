"""Joint model-recipe x B search. Dry-run by default; use --execute to train."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import datetime
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback

import torch

from cf.data import CFData, DATASETS, file_hash
from cf.models import MODEL_FAMILIES, model_variants, source_hashes
from train_cf import ROOT, parse_args as train_args, run_one


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def representations(spec):
    values = spec['degree_values']
    weighted = spec['weighted_mean_values']
    if (not values or len(set(values)) != len(values) or len(set(weighted)) != len(weighted)
            or any(not math.isfinite(x) or x < 0 for x in values)
            or any(not math.isfinite(x) or x <= 0 for x in weighted)):
        raise ValueError('Invalid or duplicate B exponents; weighted b=0 duplicates mean')
    presets = {(0., 0.): 'sum', (1., 0.): 'mean', (.5, 0.): 'sqrt', (.5, .5): 'sym'}
    result = []
    for a, b in itertools.product(values, repeat=2):
        mode = presets.get((a, b), 'degree')
        label = mode if mode != 'degree' else 'degree_a{:g}_b{:g}'.format(a, b).replace('.', 'p')
        result.append(dict(id=label, params={'b-mode': mode, 'b-a': a, 'b-b': b}))
    for b in weighted:
        result.append(dict(id='weighted_mean_b{:g}'.format(b).replace('.', 'p'),
                           params={'b-mode': 'weighted_mean', 'b-a': 0., 'b-b': b}))
    return result


def build_jobs(spec, datasets, models, phase, seeds):
    allowed = {'epochs', 'eval-every', 'patience', 'top-k', 'dim', 'context-hops', 'batch-size',
               'lr', 'l2', 'weight-decay', 'pool', 'encoder', 'n-negs', 'window-length',
               'alpha', 'ssl-temp', 'cl-rate', 'eps', 'gamma',
               'fism-alpha', 'fism-reg', 'fism-negatives', 'fism-user-bias'}
    b_trials = representations(spec['representations'])
    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for family in models:
                model_spec = spec['models'][family]
                grid = model_spec['grid']
                if any(not isinstance(v, list) or not v or len(set(v)) != len(v) for v in grid.values()):
                    raise ValueError('Grid values must be nonempty, unique lists')
                for recipe_no, values in enumerate(itertools.product(*grid.values())):
                    params = dict(spec['common'], **model_spec['fixed'])
                    params.update(zip(grid, values))
                    if set(params) - allowed:
                        raise ValueError('Unsupported model parameter keys: ' + str(set(params) - allowed))
                    for variant in model_variants(family, phase):
                        reps = b_trials if variant == 'item-only' else [dict(id='none', params={})]
                        for rep in reps:
                            job = dict(dataset=dataset, seed=seed, model=family, variant=variant,
                                       recipe=recipe_no, B=rep['id'], params=dict(params, **rep['params']))
                            job['id'] = '{}_{}_s{}_r{:03d}_{}_{}'.format(
                                family, variant, seed, recipe_no, rep['id'], digest(job)[:12])
                            jobs.append(job)
    if len({(j['dataset'], j['id']) for j in jobs}) != len(jobs):
        raise ValueError('Duplicate search jobs')
    return jobs


def job_args(job, options):
    argv = ['--dataset', job['dataset'], '--model', job['model'], '--variant', job['variant'],
            '--seed', str(job['seed']), '--data-root', str(options.data_root),
            '--itemgcn-root', str(options.itemgcn_root), '--device', options.device,
            '--eval-batch-size', str(options.eval_batch_size), '--threads', str(options.threads),
            '--duplicate-items', options.duplicate_items,
            '--log-every', str(options.log_every if options.log_every is not None else 0)]
    if not options.test_after_train:
        argv += ['--validation-only']
    if options.delete_checkpoint_after_test:
        argv += ['--delete-checkpoint-after-test']
    params = dict(job['params'])
    if options.smoke:
        params.update({'epochs': 1, 'batch-size': 32})
        argv += ['--max-steps', '1', '--eval-users', '32']
    for key, value in params.items():
        argv += ['--' + key, str(value)]
    return train_args(argv)


def freeze_manifest(root, manifest):
    path = root / 'manifest.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Manifest mismatch: config/data/code/runtime changed. Use a NEW output directory.')
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'manifest.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError('Another worker is initializing the manifest; retry shortly: ' + str(lock))
    try:
        with handle:
            handle.write('{} {}'.format(socket.gethostname(), os.getpid()))
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError('Manifest mismatch; use a NEW output directory')
        else:
            atomic_json(path, manifest)
    finally:
        lock.unlink()


def job_directory(root, job):
    return root / job['dataset'] / job['id']


def completed_result(root, job):
    directory = job_directory(root, job)
    status_path = directory / 'status.json'
    if not status_path.exists():
        return None
    status = read_json(status_path)
    if status.get('state') != 'completed' or status.get('job_digest') != digest(job):
        return None
    run_dir = directory / status['attempt'] / 'run'
    if not (run_dir / 'result.json').is_file():
        return None
    result = read_json(run_dir / 'result.json')
    if result.get('validation_only'):
        if result.get('test') is not None:
            raise ValueError('Validation-only result unexpectedly contains test metrics: ' + str(run_dir))
    else:
        metrics = result.get('test')
        if not isinstance(metrics, dict) or not all(
                isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k])
                for k in ('recall@20', 'ndcg@20', 'hr@20', 'users')) or metrics['users'] <= 0:
            raise ValueError('Missing or invalid test metrics: ' + str(run_dir))
    if not (run_dir / 'best.pt').is_file():
        if (result.get('checkpoint_retention') != 'delete_after_test'
                or result.get('validation_only') or not result.get('test')
                or not result.get('checkpoint_sha256') or not (run_dir / 'training.txt').is_file()):
            return None
    if any(result[key] != job[key] for key in ('dataset', 'model', 'variant', 'seed')):
        raise ValueError('Search result does not match job: ' + str(run_dir))
    if not math.isfinite(result['valid']['recall@20']):
        raise ValueError('Invalid validation score: ' + str(run_dir))
    return result


def summarize(root, jobs):
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['dataset'], job['model'], job['variant'], job['seed'])].append(job)
    rows = []
    for (dataset, model, variant, seed), group in groups.items():
        results = [(job, completed_result(root, job)) for job in group]
        done = [(job, result) for job, result in results if result is not None]
        ranked = [(job, result) for job, result in done if not result['smoke_test']]
        best = max(ranked, key=lambda pair: pair[1]['valid']['recall@20']) if ranked else None
        rows.append(dict(dataset=dataset, model=model, variant=variant, seed=seed,
                         planned=len(group), completed=len(done),
                         complete=len(ranked) == len(group),
                         smoke_results=sum(result['smoke_test'] for _, result in done),
                         best=None if best is None else dict(job=best[0], result=best[1]),
                         selection='validation Recall@20; incomplete groups are provisional'))
    return rows


def execute_job(root, job, args, data, graph, device, retry_failed=False):
    directory = job_directory(root, job)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'running.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        return 'busy'
    try:
        with handle:
            json.dump(dict(host=socket.gethostname(), pid=os.getpid()), handle)
        if completed_result(root, job) is not None:
            return 'skipped'
        status_path = directory / 'status.json'
        if status_path.exists() and read_json(status_path).get('state') == 'failed' and not retry_failed:
            return 'failed-skipped'
        attempt = 'attempt_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        attempt_dir = directory / attempt
        attempt_dir.mkdir(exist_ok=False)
        run_dir = attempt_dir / 'run'
        status = dict(state='running', attempt=attempt, job_digest=digest(job),
                      host=socket.gethostname(), pid=os.getpid(), device=str(device))
        atomic_json(status_path, status)
        atomic_json(attempt_dir / 'job.json', job)
        try:
            with (attempt_dir / 'train.log').open('w', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log):
                    b = (data.aggregation(args.b_mode, args.b_a, args.b_b)
                         if job['variant'] == 'item-only' and job['model'] != 'fism' else None)
                    run_one(data, args, job['model'], job['variant'], device, graph, b,
                            run_dir=run_dir)
            status['state'] = 'completed'
            atomic_json(status_path, status)
            return 'completed'
        except Exception:
            status.update(state='failed', error=traceback.format_exc())
            atomic_json(status_path, status)
            print(status['error'], file=sys.stderr, flush=True)
            return 'failed'
        finally:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        lock.unlink()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/cf_joint_search.json')
    p.add_argument('--datasets', nargs='+', default=['Movie'])
    p.add_argument('--models', nargs='+', choices=list(MODEL_FAMILIES) + ['all'], default=['all'])
    p.add_argument('--phase', choices=['original', 'item-only', 'both', 'native'], default='both')
    p.add_argument('--seeds', nargs='+', type=int, default=[2025])
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--duplicate-items', choices=['error', 'count'], default='error')
    p.add_argument('--itemgcn-root', type=Path, default=ROOT.parent / 'ItemGCN')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--test-after-train', action='store_true', help='Test each validation-selected checkpoint immediately')
    p.add_argument('--delete-checkpoint-after-test', action='store_true', help='Remove each checkpoint after saving its test results')
    p.add_argument('--log-every', type=int, default=None, help='Step log interval; 0 disables step logs')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--max-jobs', type=int, help='Attempt cap for this invocation; 0 means unlimited')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--prepare', action='store_true', help='Validate files and freeze manifest without training')
    p.add_argument('--smoke', action='store_true', help='One step, 32 eval users; separate default output, no best selection')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--summarize', action='store_true', help='Read frozen manifest and statuses; no training')
    args = p.parse_args(argv)
    if sum((args.execute, args.prepare, args.summarize)) > 1:
        p.error('Choose execute, prepare, OR summarize')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error('Require 0 <= shard-index < num-shards')
    if args.threads < 1 or args.eval_batch_size < 1 or (args.max_jobs is not None and args.max_jobs < 0):
        p.error('Invalid threads/batch-size/max-jobs')
    if args.log_every is not None and args.log_every < 0:
        p.error('log-every must be nonnegative')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        p.error('Seeds must be unique uint32 values')
    if 'all' in args.models and len(args.models) != 1:
        p.error('all cannot be combined with named models')
    if len(set(args.models)) != len(args.models):
        p.error('Duplicate models')
    args.models = [m for m in MODEL_FAMILIES if args.models == ['all'] or m in args.models]
    if args.datasets == ['all']:
        args.datasets = list(DATASETS.values())
    else:
        args.datasets = [DATASETS.get(d.lower(), d) for d in args.datasets]
    if len(set(args.datasets)) != len(args.datasets) or any(d not in DATASETS.values() for d in args.datasets):
        p.error('Unknown or duplicate datasets')
    if args.output is None:
        args.output = ROOT / 'weights' / ('cf_joint_smoke' if args.smoke else 'cf_joint_search')
    args.output = args.output.resolve()
    if args.max_jobs is None:
        args.max_jobs = 1 if args.smoke else 0
    return args


def main(argv=None):
    options = parse_args(argv)
    root = options.output
    if options.summarize:
        manifest = read_json(root / 'manifest.json')
        report = summarize(root, manifest['jobs'])
        atomic_json(root / 'summary.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    spec = read_json(options.config)
    execution = spec.get('execution', {})
    if set(execution) - {'test_after_train', 'delete_checkpoint_after_test', 'log_every'}:
        raise ValueError('Unknown execution settings')
    for name in ('test_after_train', 'delete_checkpoint_after_test'):
        if name in execution and not isinstance(execution[name], bool):
            raise ValueError(name + ' must be boolean')
        setattr(options, name, getattr(options, name) or execution.get(name, False))
    if options.log_every is None:
        options.log_every = execution.get('log_every', 0)
    if type(options.log_every) is not int or options.log_every < 0:
        raise ValueError('log_every must be a nonnegative integer')
    if options.delete_checkpoint_after_test and not options.test_after_train:
        raise ValueError('Deleting checkpoints requires test_after_train')
    if spec.get('version') != 1 or spec['common'].get('top-k') != 20:
        raise ValueError('Search requires config version 1 and top-k=20')
    jobs = build_jobs(spec, options.datasets, options.models, options.phase, options.seeds)
    selected = jobs[options.shard_index::options.num_shards]
    if not selected:
        raise ValueError('Empty shard: reduce num-shards or select more jobs')
    for (dataset, model, variant), count in Counter(
            (j['dataset'], j['model'], j['variant']) for j in jobs).items():
        print('{} / {} / {}: {}'.format(dataset, model, variant, count))
    print('Total={} shard={}/{} jobs={} output={}'.format(
        len(jobs), options.shard_index, options.num_shards, len(selected), root), flush=True)
    # Parse every command even on dry-run; do not silently freeze an invalid grid.
    for job in jobs:
        job_args(job, options)
    if not options.execute and not options.prepare:
        print('DRY RUN: no training and no output files written. Add --execute to run.', flush=True)
        return jobs
    device = torch.device(options.device)
    if options.execute and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU PyTorch environment or --device cpu --smoke')
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Only CPU/CUDA are supported')
    fingerprints = {}
    for dataset in options.datasets:
        fingerprints[dataset] = {split: file_hash(options.data_root / dataset / 'cf_data' / (split + '.txt'))
                                 for split in ('train', 'valid', 'test')}
    manifest = dict(version=1, jobs=jobs, config=spec, data_sha256=fingerprints,
                    source_sha256=source_hashes(options.itemgcn_root) if any(j['model'] != 'fism' for j in jobs) else {},
                    adapter_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in
                                    [ROOT / 'train_cf.py', Path(__file__).resolve()] + sorted((ROOT / 'cf').glob('*.py'))},
                    runtime=dict(torch=str(torch.__version__), python=sys.version.split()[0],
                                 eval_batch_size=options.eval_batch_size, threads=options.threads,
                                 smoke=options.smoke, duplicate_items=options.duplicate_items,
                                 test_after_train=options.test_after_train,
                                 delete_checkpoint_after_test=options.delete_checkpoint_after_test,
                                 log_every=options.log_every))
    freeze_manifest(root, manifest)
    if options.prepare:
        print('Prepared manifest; no training launched: ' + str(root / 'manifest.json'), flush=True)
        return manifest
    torch.set_num_threads(options.threads)
    active_dataset, data, graph = None, None, None
    counts, attempts = Counter(), 0
    for job in selected:
        if options.max_jobs and attempts >= options.max_jobs:
            break
        if completed_result(root, job) is not None:
            counts['skipped'] += 1
            continue
        if active_dataset != job['dataset']:
            data = CFData(options.data_root / job['dataset'] / 'cf_data', options.duplicate_items)
            graph = data.graph() if any(j['model'] != 'fism' for j in selected) else None
            active_dataset = job['dataset']
        args = job_args(job, options)
        print('START {} / {}'.format(job['dataset'], job['id']), flush=True)
        state = execute_job(root, job, args, data, graph, device, options.retry_failed)
        counts[state] += 1
        if state in ('completed', 'failed'):
            attempts += 1
        print('{} {}'.format(state.upper(), job['id']), flush=True)
    report = summarize(root, jobs)
    atomic_json(root / 'summary_shard_{}_of_{}.json'.format(options.shard_index, options.num_shards), report)
    print('Worker counts: ' + json.dumps(counts), flush=True)
    if counts['failed'] or counts['failed-skipped']:
        raise RuntimeError('Some jobs failed. Inspect status.json / train.log; rerun with --retry-failed after fixing.')
    return report


if __name__ == '__main__':
    main()
