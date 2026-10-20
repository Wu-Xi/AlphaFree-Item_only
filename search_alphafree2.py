"""29-B AlphaFree2 search; dry-run by default, sequential training on one GPU."""
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

from models.base.parse import parse_args as model_args
from models.base.user_aggregation import aggregation_grid

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


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
    options = parser.parse_args(argv)
    if options.max_runs < 0 or options.cuda < 0 or not 0 <= options.seed < 2**32:
        parser.error('Invalid run limit, CUDA index or seed')
    options.data_root = options.data_root.resolve()
    options.output = (options.output or ROOT / 'weights' /
                      ('alphafree2_search_' + options.dataset + '_s' + str(options.seed))).resolve()
    return options


def build_jobs(options):
    jobs = []
    for setting in aggregation_grid():
        argv = ['--phase', 'train', '--model_name', 'AlphaFree2', '--dataset', options.dataset,
                '--config_dir', str(ROOT / 'configs'), '--data_path', str(options.data_root) + os.sep,
                '--seed', str(options.seed), '--validation-only', '--saveID', 'B29_' + setting['id'],
                '--b-mode', setting['mode'], '--b-a', str(setting['a']), '--b-b', str(setting['b'])]
        for key in ('max_epoch', 'batch_size', 'lr', 'num_workers'):
            value = getattr(options, key)
            if value is not None:
                argv.extend(['--' + key, str(value)])
        args, _ = model_args(argv)
        jobs.append(dict(id=setting['id'], argv=argv, parameters=vars(args)))
    return jobs


def input_fingerprints(options, args):
    dataset = options.data_root / options.dataset
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
    return {str(path): file_hash(path) for path in files}


def completed(root, job):
    path = root / (job['id'] + '.json')
    if not path.is_file():
        return None
    status = read_json(path)
    if status.get('state') != 'completed':
        return None
    result = status['result']
    args = job['parameters']
    if (not result['validation_only'] or result['model'] != 'AlphaFree2'
            or any(result[k] != args[k] for k in ('dataset', 'seed', 'b_mode', 'b_a', 'b_b'))
            or not math.isfinite(result['best_valid_recall'])
            or not Path(result['checkpoint']).is_file()):
        raise ValueError('Invalid completed result: ' + str(path))
    return result


def summary(root, jobs):
    results = [dict(id=job['id'], result=completed(root, job)) for job in jobs]
    done = [row for row in results if row['result'] is not None]
    return dict(planned=len(jobs), completed=len(done), complete=len(done) == len(jobs),
                selection='validation Recall@20 only; incomplete best is provisional',
                best=max(done, key=lambda row: row['result']['best_valid_recall']) if done else None,
                results=results)


def main(argv=None):
    options = parse_args(argv)
    jobs = build_jobs(options)
    print('AlphaFree2: 29 B settings; ORIGINAL user branch only; augmented branch unchanged.')
    print('Other hyperparameters: dataset YAML + original defaults + explicit training overrides.')
    for job in jobs:
        command = [sys.executable, str(ROOT / 'main.py')] + job['argv'] + ['--cuda', str(options.cuda)]
        print(job['id'] + ': ' + shlex.join(command))
    if not options.execute:
        print('DRY RUN: no training or files written. Add --execute on a configured GPU server.')
        return jobs
    import torch
    if not torch.cuda.is_available() or options.cuda >= torch.cuda.device_count():
        raise RuntimeError('AlphaFree2 training requires a valid CUDA device')
    fingerprints = input_fingerprints(options, jobs[0]['parameters'])
    sources = [ROOT / 'main.py', Path(__file__).resolve()] + sorted((ROOT / 'models').rglob('*.py'))
    manifest = dict(version=1, jobs=jobs, data_sha256=fingerprints,
                    source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                    config_sha256=file_hash(ROOT / 'configs' / (options.dataset + '.yaml')),
                    python=sys.version.split()[0], torch=str(torch.__version__))
    options.output.mkdir(parents=True, exist_ok=True)
    lock = options.output / 'running.lock'
    # Only one worker per output. Different datasets/experiments may use different outputs.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest_path = options.output / 'manifest.json'
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError('Code/data/config/runtime changed: use a NEW output directory')
        else:
            write_json(manifest_path, manifest)
        attempts = 0
        for job in jobs:
            if completed(options.output, job) is not None:
                print('SKIP completed: ' + job['id'], flush=True)
                continue
            if options.max_runs and attempts >= options.max_runs:
                break
            status_path = options.output / (job['id'] + '.json')
            old_status = read_json(status_path) if status_path.exists() else {}
            if old_status.get('state') in ('failed', 'running') and not options.retry_failed:
                raise RuntimeError('Failed/interrupted run; inspect log and use --retry-failed: ' + job['id'])
            attempt = int(old_status.get('attempt', 0)) + 1
            log_path = options.output / (job['id'] + '.attempt' + str(attempt) + '.log')
            status = dict(state='running', attempt=attempt, log=str(log_path))
            write_json(status_path, status)
            run_args = list(job['argv'])
            save_index = run_args.index('--saveID') + 1
            run_args[save_index] += '_' + uuid.uuid4().hex[:10]
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
                status.update(state='completed', result=result)
                write_json(status_path, status)
                completed(options.output, job)  # Verify metadata and checkpoint before accepting it.
            except Exception as error:
                status.update(state='failed', error=str(error))
                write_json(status_path, status)
                raise
            attempts += 1
            write_json(options.output / 'summary.json', summary(options.output, jobs))
        report = summary(options.output, jobs)
        write_json(options.output / 'summary.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
