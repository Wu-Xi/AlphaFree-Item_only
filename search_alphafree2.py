"""AlphaFree original/mean + 28 AlphaFree2 variants, with independent GPU shards."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
import tempfile
from contextlib import contextmanager

from models.base.parse import parse_args as model_args
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


@contextmanager
def exclusive_lock(path):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        Path(path).unlink()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='amazon_movie',
                        choices=['amazon_movie', 'amazon_book_2014', 'amazon_video', 'amazon_baby',
                                 'steam', 'amazon_beauty_personal', 'amazon_health'])
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', help='Optional multi-dataset plan, e.g. Baby Steam')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-runs', type=int, default=0, help='0 means all remaining B settings')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='Check caches and freeze plan, no training')
    parser.add_argument('--summarize', action='store_true', help='Read frozen plan and select by validation')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    options = parser.parse_args(argv)
    if options.max_runs < 0 or options.cuda < 0 or not 0 <= options.seed < 2**32:
        parser.error('Invalid run limit, CUDA index or seed')
    if options.num_shards < 1 or not 0 <= options.shard_index < options.num_shards:
        parser.error('Require 0 <= shard-index < num-shards')
    if sum((options.execute, options.prepare, options.summarize)) > 1:
        parser.error('Choose execute, prepare, or summarize')
    aliases = dict(movie='amazon_movie', book='amazon_book_2014', video='amazon_video',
                   baby='amazon_baby', steam='steam', beauty='amazon_beauty_personal', health='amazon_health')
    options.datasets = [aliases.get(d.lower(), d) for d in (options.datasets or [options.dataset])]
    if len(set(options.datasets)) != len(options.datasets) or any(d not in aliases.values() for d in options.datasets):
        parser.error('Unknown or duplicate datasets')
    options.data_root = options.data_root.resolve()
    options.output = (options.output or ROOT / 'weights' /
                      ('alpha_search_' + '_'.join(options.datasets) + '_s' + str(options.seed))).resolve()
    return options


def build_jobs(options):
    jobs = []
    for dataset in options.datasets:
        for setting in sorted(aggregation_grid(), key=lambda s: s['id'] != 'mean'):
            model = 'AlphaFree' if setting['id'] == 'mean' else 'AlphaFree2'
            argv = ['--phase', 'train', '--model_name', model, '--dataset', dataset,
                    '--config_dir', str(ROOT / 'configs'), '--data_path', str(options.data_root) + os.sep,
                    '--seed', str(options.seed), '--validation-only', '--saveID', 'B29_' + setting['id'],
                    '--b-mode', setting['mode'], '--b-a', str(setting['a']), '--b-b', str(setting['b'])]
            for key in ('max_epoch', 'batch_size', 'lr', 'num_workers'):
                value = getattr(options, key)
                if value is not None:
                    argv.extend(['--' + key, str(value)])
            args, _ = model_args(argv)
            if args.Ks != 20:
                raise ValueError('This search selects by validation Recall@20; require Ks=20')
            jobs.append(dict(id=dataset + '__' + setting['id'], setting=setting,
                             variant='original' if model == 'AlphaFree' else 'item-only',
                             argv=argv, parameters=vars(args)))
    return jobs


def input_fingerprints(options, args):
    dataset = options.data_root / args['dataset']
    language_file = ('item_cf_embeds_large3_array.npy' if args['lm_model'] == 'v3'
                     else 'item_cf_embeds_LLAMA_array.npy')
    # Require complete caches: never trigger expensive automatic preprocessing in a grid.
    files = [dataset / 'cf_data' / (split + '.txt') for split in ('train', 'valid', 'test')]
    files += [dataset / 'item_info' / language_file]
    files += [dataset / 'preprocessed' / name for name in
              ('user_interact_matrix.npz', 'aug_user_interact_matrix.npz', 'item_cf_embeds_augmented.npy')]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError('Prepare data/cache before search: ' + str(path))
    shape = validate_mean_cache(dataset)
    import numpy as np
    from scipy import sparse
    original_features = np.load(files[3], mmap_mode='r', allow_pickle=False)
    augmented_features = np.load(files[6], mmap_mode='r', allow_pickle=False)
    augmented_users = sparse.load_npz(files[5])
    if (original_features.ndim != 2 or original_features.shape[0] != shape[1]
            or augmented_features.shape != original_features.shape
            or augmented_users.shape != shape or not np.isfinite(augmented_users.data).all()):
        raise ValueError('Inconsistent feature/augmentation cache shapes: ' + str(dataset))
    return {str(path): file_hash(path) for path in files}


def validate_mean_cache(dataset):
    """Verify the original cached user matrix matches counted TRAIN history."""
    import numpy as np
    from scipy import sparse
    from models.base.user_aggregation import build_user_aggregation
    history, all_users, all_items = {}, set(), set()
    for split in ('train', 'valid', 'test'):
        seen = set()
        with (dataset / 'cf_data' / (split + '.txt')).open(encoding='utf-8-sig') as stream:
            for line in stream:
                values = [int(v) for v in line.split()]
                if not values:
                    continue
                user, items = values[0], values[1:]
                if min(values) < 0 or user in seen:
                    raise ValueError('Negative ID or duplicate user row in ' + str(dataset) + '/' + split)
                seen.add(user)
                all_users.add(user)
                all_items.update(items)
                if split == 'train':
                    history[user] = items
    original = sparse.load_npz(dataset / 'preprocessed/user_interact_matrix.npz').tocsr()
    if all_users != set(range(original.shape[0])) or all_items != set(range(original.shape[1])):
        raise ValueError('Catalog IDs do not match cached matrix dimensions: ' + str(dataset))
    rebuilt = build_user_aggregation(history, *original.shape, mode='mean', preserve_counts=True)
    difference = original - rebuilt
    if (not np.isfinite(original.data).all() or
            (difference.nnz and np.abs(difference.data).max() > 1e-6)):
        raise ValueError('Original mean cache does not match counted training history: ' + str(dataset))
    return original.shape


def completed(root, job):
    path = root / (job['id'] + '.json')
    if not path.is_file():
        return None
    status = read_json(path)
    if status.get('state') != 'completed':
        return None
    result = status['result']
    validate_result(job, result)
    return result


def validate_result(job, result):
    args = job['parameters']
    if (not result['validation_only'] or result['model'] != args['model_name']
            or any(result[k] != args[k] for k in ('dataset', 'seed', 'b_mode', 'b_a', 'b_b'))
            or not math.isfinite(result['best_valid_recall'])
            or not Path(result['checkpoint']).is_file()):
        raise ValueError('Invalid completed result: ' + job['id'])


def summary(root, jobs):
    results = [dict(id=job['id'], result=completed(root, job)) for job in jobs]
    done = [row for row in results if row['result'] is not None]
    groups = []
    for dataset in dict.fromkeys(j['parameters']['dataset'] for j in jobs):
        group_jobs = [j for j in jobs if j['parameters']['dataset'] == dataset]
        group_done = [row for row in done if row['result']['dataset'] == dataset]
        variants = [row for row in group_done if row['result']['model'] == 'AlphaFree2']
        original = next((row for row in group_done if row['result']['model'] == 'AlphaFree'), None)
        groups.append(dict(dataset=dataset, planned=len(group_jobs), completed=len(group_done),
                           complete=len(group_done) == len(group_jobs), original=original,
                           best_variant=max(variants, key=lambda row: row['result']['best_valid_recall']) if variants else None))
    return dict(planned=len(jobs), completed=len(done), complete=len(done) == len(jobs),
                selection='validation Recall@20 only; select variants separately within each dataset',
                groups=groups, results=results)


def make_manifest(options, jobs):
    import torch
    fingerprints, configs = {}, {}
    for dataset in options.datasets:
        args = next(j['parameters'] for j in jobs if j['parameters']['dataset'] == dataset)
        fingerprints.update(input_fingerprints(options, args))
        configs[dataset] = file_hash(ROOT / 'configs' / (dataset + '.yaml'))
    sources = [ROOT / 'main.py', Path(__file__).resolve(), ROOT / 'eval_alpha_experiment.py',
               ROOT / 'cf/data.py', ROOT / 'cf/evaluate.py'] + sorted((ROOT / 'models').rglob('*.py'))
    return dict(version=2, protocol='original + 28 variants; counted train history; validation only',
                jobs=jobs, data_sha256=fingerprints,
                source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                config_sha256=configs, python=sys.version.split()[0], torch=str(torch.__version__))


def freeze_manifest(root, manifest):
    path = root / 'manifest.json'
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError('Code/data/config/runtime changed: use a NEW output directory')
        return
    with exclusive_lock(root / 'manifest.lock'):
        if path.exists() and read_json(path) != manifest:
            raise ValueError('Manifest changed: use a NEW output directory')
        write_json(path, manifest)


def save_report(root, jobs, filename):
    report = summary(root, jobs)
    write_json(root / filename, report)
    return report


def execute_job(options, job):
    # Lock before reading/changing status; protects even overlapping shard plans.
    with exclusive_lock(options.output / (job['id'] + '.lock')):
        if completed(options.output, job) is not None:
            return False
        status_path = options.output / (job['id'] + '.json')
        old = read_json(status_path) if status_path.exists() else {}
        if old.get('state') in ('failed', 'running') and not options.retry_failed:
            raise RuntimeError('Failed/interrupted run; inspect log and use --retry-failed: ' + job['id'])
        attempt = int(old.get('attempt', 0)) + 1
        log_path = options.output / (job['id'] + '.attempt' + str(attempt) + '.log')
        status = dict(state='running', attempt=attempt, log=str(log_path), pid=os.getpid())
        write_json(status_path, status)
        run_args = list(job['argv'])
        run_args[run_args.index('--saveID') + 1] += '_' + uuid.uuid4().hex[:10]
        command = [sys.executable, '-u', str(ROOT / 'main.py')] + run_args + ['--cuda', str(options.cuda)]
        print('START ' + job['id'] + ' -> ' + str(log_path), flush=True)
        try:
            with log_path.open('x', encoding='utf-8') as log:
                process = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError('Training exit code ' + str(process.returncode))
            result = None
            with log_path.open(encoding='utf-8', errors='replace') as log:
                for line in log:
                    if line.startswith('ALPHAFREE2_RESULT '):
                        result = json.loads(line[len('ALPHAFREE2_RESULT '):])
            if result is None:
                raise RuntimeError('Training did not produce a completed result')
            validate_result(job, result)
            status.update(state='completed', result=result, checkpoint_sha256=file_hash(result['checkpoint']))
            write_json(status_path, status)
        except Exception as error:
            status.update(state='failed', error=str(error))
            write_json(status_path, status)
            raise
        return True


def main(argv=None):
    options = parse_args(argv)
    if options.summarize:
        manifest = read_json(options.output / 'manifest.json')
        if manifest.get('version') != 2:
            raise ValueError('Expected original + 28 manifest version 2')
        report = save_report(options.output, manifest['jobs'], 'summary.json')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    jobs = build_jobs(options)
    selected = jobs[options.shard_index::options.num_shards]
    if not selected:
        raise ValueError('Empty shard; reduce num-shards')
    print('AlphaFree: 1 original/mean + 28 variants per dataset. Total={}, shard={}/{}, jobs={}'.format(
        len(jobs), options.shard_index, options.num_shards, len(selected)))
    print('Other hyperparameters: dataset YAML + original defaults + explicit training overrides.')
    for job in selected:
        command = [sys.executable, str(ROOT / 'main.py')] + job['argv'] + ['--cuda', str(options.cuda)]
        print(job['id'] + ': ' + shlex.join(command))
    if not options.execute and not options.prepare:
        print('DRY RUN: no training or files written. Add --execute on a configured GPU server.')
        return jobs
    import torch
    if options.execute and (not torch.cuda.is_available() or options.cuda >= torch.cuda.device_count()):
        raise RuntimeError('AlphaFree2 training requires a valid CUDA device')
    manifest = make_manifest(options, jobs)
    options.output.mkdir(parents=True, exist_ok=True)
    freeze_manifest(options.output, manifest)
    if options.prepare:
        print('Prepared; no training launched.')
        return manifest
    with exclusive_lock(options.output / ('shard_{}_of_{}.lock'.format(options.shard_index, options.num_shards))):
        attempts = 0
        for job in selected:
            if completed(options.output, job) is not None:
                print('SKIP completed: ' + job['id'], flush=True)
                continue
            if options.max_runs and attempts >= options.max_runs:
                break
            attempts += int(execute_job(options, job))
            save_report(options.output, jobs, 'summary_shard_{}_of_{}.json'.format(options.shard_index, options.num_shards))
        report = save_report(options.output, jobs, 'summary_shard_{}_of_{}.json'.format(options.shard_index, options.num_shards))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


if __name__ == '__main__':
    main()
