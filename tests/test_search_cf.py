from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cf.data import CFData
from search_cf import (ROOT, atomic_json, build_jobs, completed_result, digest, execute_job,
                       freeze_manifest, job_args, job_directory, main, parse_args, read_json,
                       representations, summarize)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = read_json(ROOT / 'configs/cf_joint_search.json')
        self.jobs = build_jobs(self.spec, ['amazon_movie'], list(self.spec['models']), 'both', [2025])
        self.options = parse_args(['--output', str(self.root / 'search'), '--device', 'cpu', '--smoke'])
        directory = self.root / 'data/amazon_movie/cf_data'
        directory.mkdir(parents=True)
        for name, content in dict(train='0 0 1\n1 1 2\n2 2 3\n',
                                  valid='0 2\n1 3\n2 4\n', test='0 3\n1 4\n2 0\n').items():
            (directory / (name + '.txt')).write_text(content, encoding='utf-8')
        self.options.data_root = self.root / 'data'
        self.data = CFData(directory)
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_780_jobs_and_29_unique_B(self):
        self.assertEqual(len(self.jobs), 780)
        counts = Counter((j['model'], j['variant']) for j in self.jobs)
        self.assertEqual(counts['fism', 'original'], 1)
        self.assertEqual(counts['fism', 'item-only'], 29)
        for model, recipes in [('lightgcn-rns', 1), ('stabcf', 12), ('simgcl', 6), ('directau', 6)]:
            self.assertEqual(counts[model, 'original'], recipes)
            self.assertEqual(counts[model, 'item-only'], recipes * 29)
        reps = representations(self.spec['representations'])
        self.assertEqual(len(reps), 29)
        self.assertEqual(len({r['id'] for r in reps}), 29)
        self.assertEqual(sum(r['id'] == 'sym' for r in reps), 1)
        matrices = [self.data.aggregation(r['params']['b-mode'], r['params']['b-a'], r['params']['b-b']) for r in reps]
        self.assertTrue(all(m.shape == (3, 5) for m in matrices))

    def test_exact_legacy_model_parameter_grid(self):
        old = read_json(ROOT.parent / 'ItemGCN/experiments.json')
        for name, spec in self.spec['models'].items():
            if name == 'fism':
                continue  # New native baseline, not a legacy ItemGCN model.
            original = old['models']['rns' if name == 'lightgcn-rns' else name]
            expected_grid = {key.replace('_', '-'): value for key, value in original.get('grid', {}).items()}
            self.assertEqual(spec['grid'], expected_grid)
            for key in ('dim', 'lr', 'l2', 'weight_decay', 'context_hops', 'batch_size'):
                expected = original.get('shared', {}).get(key, old['common'][key])
                actual = spec['fixed'].get(key.replace('_', '-'), self.spec['common'][key.replace('_', '-')])
                self.assertEqual(actual, expected)
        for job in self.jobs:
            args = job_args(job, self.options)
            self.assertTrue(args.validation_only)
            if job['model'] == 'directau':
                self.assertEqual(args.weight_decay, .001)
                self.assertEqual(args.l2, 0.)

    def test_shards_partition_all_jobs_and_seeds(self):
        parts = [self.jobs[i::8] for i in range(8)]
        ids = [job['id'] for part in parts for job in part]
        self.assertEqual(len(ids), 780)
        self.assertEqual(len(set(ids)), 780)
        multi = build_jobs(self.spec, ['amazon_movie', 'steam'], ['stabcf'], 'both', [2025, 2026])
        self.assertEqual(len(multi), 12 * 30 * 2 * 2)

    def test_freeze_refuses_changed_plan(self):
        root = self.options.output
        freeze_manifest(root, {'jobs': self.jobs})
        freeze_manifest(root, {'jobs': self.jobs})
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            freeze_manifest(root, {'jobs': self.jobs[:1]})

    def test_count_policy_reaches_loader_and_is_frozen(self):
        path = self.options.data_root / 'amazon_movie/cf_data/train.txt'
        path.write_text('0 0 1 0\n1 1 2\n2 2 3\n', encoding='utf-8')
        argv = ['--datasets', 'Movie', '--models', 'fism', '--device', 'cpu',
                '--smoke', '--duplicate-items', 'count', '--data-root', str(self.options.data_root),
                '--output', str(self.options.output)]
        manifest = main(argv + ['--prepare'])
        self.assertEqual(manifest['runtime']['duplicate_items'], 'count')
        main(argv + ['--execute'])
        result = completed_result(self.options.output, manifest['jobs'][0])
        self.assertIsNotNone(result)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            main(argv + ['--duplicate-items', 'error', '--prepare'])

    def test_training_resume_and_smoke_never_selected_best(self):
        job = self.jobs[1]  # RNS item-only, first B.
        args = job_args(job, self.options)
        state = execute_job(self.options.output, job, args, self.data, self.data.graph(), torch.device('cpu'))
        self.assertEqual(state, 'completed')
        result = completed_result(self.options.output, job)
        self.assertIsNone(result['test'])
        self.assertTrue(result['smoke_test'])
        state = execute_job(self.options.output, job, args, self.data, self.data.graph(), torch.device('cpu'))
        self.assertEqual(state, 'skipped')
        summary = summarize(self.options.output, [job])[0]
        self.assertEqual(summary['completed'], 1)
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['best'])

    def test_failed_jobs_require_explicit_retry_and_keep_attempts(self):
        job = self.jobs[0]
        args = job_args(job, self.options)
        with patch('search_cf.run_one', side_effect=RuntimeError('test failure')):
            state = execute_job(self.options.output, job, args, self.data, self.data.graph(), torch.device('cpu'))
        self.assertEqual(state, 'failed')
        state = execute_job(self.options.output, job, args, self.data, self.data.graph(), torch.device('cpu'))
        self.assertEqual(state, 'failed-skipped')
        state = execute_job(self.options.output, job, args, self.data, self.data.graph(), torch.device('cpu'), True)
        self.assertEqual(state, 'completed')
        self.assertEqual(len(list(job_directory(self.options.output, job).glob('attempt_*'))), 2)

    def test_live_lock_is_not_removed_or_run(self):
        job = self.jobs[0]
        directory = job_directory(self.options.output, job)
        directory.mkdir(parents=True)
        lock = directory / 'running.lock'
        lock.write_text('existing worker', encoding='utf-8')
        state = execute_job(self.options.output, job, job_args(job, self.options), self.data,
                            self.data.graph(), torch.device('cpu'))
        self.assertEqual(state, 'busy')
        self.assertTrue(lock.exists())

    def test_summary_uses_only_valid_and_marks_partial(self):
        group = self.jobs[:2]
        for index, job in enumerate(group):
            directory = job_directory(self.options.output, job)
            run = directory / 'attempt_fixture/run'
            run.mkdir(parents=True)
            (run / 'best.pt').touch()
            result = dict(dataset=job['dataset'], model=job['model'], variant=job['variant'], seed=job['seed'],
                          smoke_test=False, validation_only=True, test=None,
                          valid={'recall@20': .1 + index})
            atomic_json(run / 'result.json', result)
            atomic_json(directory / 'status.json', dict(state='completed', job_digest=digest(job), attempt='attempt_fixture'))
        rows = summarize(self.options.output, self.jobs[:3])
        self.assertTrue(rows[0]['complete'])
        self.assertFalse(rows[1]['complete'])
        self.assertEqual(rows[1]['best']['result']['valid']['recall@20'], 1.1)
        result['test'] = {'recall@20': 1.}
        atomic_json(run / 'result.json', result)
        with self.assertRaisesRegex(ValueError, 'test metrics'):
            completed_result(self.options.output, group[-1])

    def test_dry_run_does_not_write_or_require_gpu(self):
        root = self.root / 'dry_run'
        result = main(['--datasets', 'Movie', '--output', str(root)])
        self.assertEqual(len(result), 780)
        self.assertFalse(root.exists())

    def test_prepare_then_resume_one_shard(self):
        argv = ['--datasets', 'Movie', '--models', 'lightgcn-rns', '--device', 'cpu',
                '--smoke', '--data-root', str(self.options.data_root),
                '--output', str(self.options.output)]
        manifest = main(argv + ['--prepare'])
        self.assertEqual(len(manifest['jobs']), 30)
        self.assertFalse(any(self.options.output.rglob('best.pt')))
        main(argv + ['--num-shards', '30', '--shard-index', '0', '--execute'])
        with patch('search_cf.run_one', side_effect=AssertionError('Must skip completed job')):
            main(argv + ['--num-shards', '30', '--shard-index', '0', '--execute'])
        report = main(['--output', str(self.options.output), '--summarize'])
        self.assertEqual(sum(row['completed'] for row in report), 1)
        self.assertTrue(all(row['best'] is None for row in report))


if __name__ == '__main__':
    unittest.main()
