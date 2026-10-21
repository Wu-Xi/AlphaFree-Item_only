import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cf.data import CFData
from cf.evaluate import evaluate
from search_cf import ROOT, completed_result, main, read_json, summarize
from train_cf import Tee, load_checkpoint, parse_args, run_one, write_json


class FISMLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        directory = self.root / 'data/amazon_baby/cf_data'
        directory.mkdir(parents=True)
        for split, text in dict(train='0 0 1 0\n1 1 2\n', valid='0 2\n1 3\n',
                                test='0 0 3\n1 0\n').items():
            (directory / (split + '.txt')).write_text(text, encoding='utf-8')
        self.data = CFData(directory, 'count')
        self.args = parse_args(['--model', 'fism', '--dataset', 'Baby', '--dim', '4',
            '--epochs', '2', '--batch-size', '2', '--eval-every', '1', '--threads', '2',
            '--duplicate-items', 'count', '--delete-checkpoint-after-test'])
        self.run = self.root / 'run'
        torch.set_num_threads(2)

    def tearDown(self):
        self.temp.cleanup()

    def train(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_one(self.data, self.args, 'fism', 'original', torch.device('cpu'),
                           None, None, run_dir=self.run)

    def test_best_checkpoint_is_tested_before_deletion_and_transcript_is_complete(self):
        calls = []
        last_epoch_state = {}

        def inspect(model, data, split, k, batch_size, max_users):
            calls.append(split)
            metrics = evaluate(model, data, split, k, batch_size, max_users)
            if split == 'valid':
                metrics['recall@20'] = .9 if len(calls) == 1 else .1
                if len(calls) == 2:
                    last_epoch_state.update({n: t.clone() for n, t in model.state_dict().items()})
            else:
                self.assertEqual(max_users, 0)
                checkpoint = load_checkpoint(self.run / 'best.pt', 'cpu')
                self.assertEqual(checkpoint['epoch'], 1)
                self.assertTrue(all(torch.equal(t, checkpoint['state_dict'][n])
                                    for n, t in model.state_dict().items()))
                self.assertTrue(any(not torch.equal(t, last_epoch_state[n])
                                    for n, t in model.state_dict().items()))
            return metrics

        with patch('train_cf.evaluate', side_effect=inspect):
            result = self.train()
        self.assertEqual(calls, ['valid', 'valid', 'test'])
        self.assertEqual(result['best_epoch'], 1)
        self.assertEqual(result['test']['users'], 2)
        self.assertFalse((self.run / 'best.pt').exists())
        self.assertEqual(read_json(self.run / 'result.json'), result)
        text = (self.run / 'training.txt').read_text(encoding='utf-8')
        for token in ('CONFIG ', '"epoch": 1', '"epoch": 2', 'TEST_START best_epoch=1',
                      'TEST ', 'RESULT ', 'CHECKPOINT_DELETED'):
            self.assertIn(token, text)
        self.assertNotRegex(text, r'epoch \d+ step \d+ loss')
        self.assertEqual(len(result['checkpoint_sha256']), 64)
        self.assertEqual(result['data_sha256'], self.data.stats['sha256'])
        self.assertEqual(read_json(self.run / 'config.json')['config']['fism_reg'], 1e-6)

    def test_test_failure_keeps_checkpoint(self):
        def fail_test(model, data, split, *args):
            if split == 'test':
                raise RuntimeError('test evaluation failed')
            return evaluate(model, data, split, *args)
        with patch('train_cf.evaluate', side_effect=fail_test):
            with self.assertRaisesRegex(RuntimeError, 'test evaluation failed'):
                self.train()
        self.assertTrue((self.run / 'best.pt').is_file())
        self.assertFalse((self.run / 'result.json').exists())

    def test_result_write_failure_keeps_checkpoint(self):
        def fail_result(path, value):
            if Path(path).name == 'result.json':
                raise OSError('result disk failure')
            return write_json(path, value)
        with patch('train_cf.write_json', side_effect=fail_result):
            with self.assertRaisesRegex(OSError, 'result disk failure'):
                self.train()
        self.assertTrue((self.run / 'best.pt').is_file())
        self.assertFalse((self.run / 'result.json').exists())

    def test_transcript_write_failure_keeps_checkpoint(self):
        original_write = Tee.write
        def fail_transcript(stream, text):
            if text.startswith('RESULT '):
                raise OSError('transcript disk failure')
            return original_write(stream, text)
        with patch.object(Tee, 'write', fail_transcript):
            with self.assertRaisesRegex(OSError, 'transcript disk failure'):
                self.train()
        self.assertTrue((self.run / 'best.pt').is_file())

    def test_keep_policy_and_explicit_step_logging_still_work(self):
        self.args.delete_checkpoint_after_test = False
        self.args.log_every = 1
        result = self.train()
        self.assertTrue((self.run / 'best.pt').is_file())
        self.assertEqual(result['checkpoint_retention'], 'keep')
        self.assertRegex((self.run / 'training.txt').read_text(encoding='utf-8'), r'epoch 1 step 1 loss')

    def test_delete_without_testing_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--validation-only', '--delete-checkpoint-after-test'])

    def test_search_config_tests_deletes_and_resumes_without_checkpoint(self):
        spec = read_json(ROOT / 'configs/fism_baby_steam.json')
        spec['common'].update(epochs=2, dim=4, **{'eval-every': 1, 'batch-size': 2})
        config = self.root / 'spec.json'
        config.write_text(json.dumps(spec), encoding='utf-8')
        output = self.root / 'search'
        argv = ['--config', str(config), '--datasets', 'Baby', '--models', 'fism',
                '--phase', 'original', '--data-root', str(self.root / 'data'),
                '--duplicate-items', 'count', '--device', 'cpu', '--threads', '2',
                '--output', str(output), '--execute']
        with redirect_stdout(io.StringIO()):
            main(argv)
        manifest = read_json(output / 'manifest.json')
        self.assertTrue(manifest['runtime']['test_after_train'])
        self.assertTrue(manifest['runtime']['delete_checkpoint_after_test'])
        job = manifest['jobs'][0]
        result = completed_result(output, job)
        self.assertIsNotNone(result['test'])
        self.assertFalse(result['smoke_test'])
        self.assertFalse(list(output.rglob('best.pt')))
        self.assertTrue(summarize(output, [job])[0]['complete'])
        with patch('search_cf.run_one', side_effect=AssertionError('Must skip')), redirect_stdout(io.StringIO()):
            main(argv)
        log = next(output.rglob('training.txt'))
        log.unlink()
        self.assertIsNone(completed_result(output, job))


if __name__ == '__main__':
    unittest.main()
