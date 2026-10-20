from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from cf.data import CFData
from cf.evaluate import evaluate
from models.AlphaFree import AlphaFree
from models.AlphaFree3 import AlphaFree3, load_frozen_state
from models.base.user_aggregation import aggregation_grid
import eval_alphafree3 as runner


class AlphaFree3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        folder = self.root / 'data/amazon_movie/cf_data'
        folder.mkdir(parents=True)
        for name, content in dict(train='0 0 1\n1 1 2 3\n2 1\n3\n',
                                  valid='0 2\n1 4\n2 0\n3 2\n',
                                  test='0 3\n1 0\n2 4\n3 4\n').items():
            (folder / (name + '.txt')).write_text(content, encoding='utf-8')
        self.data = CFData(folder)
        self.features = np.random.RandomState(42).normal(size=(5, 8)).astype(np.float64)
        args = types.SimpleNamespace(model_name='AlphaFree', cuda='cpu', hidden_size=4, regs=0.,
                                     batch_size=4, tau_r=.15, tau_a=.2, lambda_align=.2, lm_model='v3')
        data = types.SimpleNamespace(n_items=5, item_cf_embeds_original=self.features,
                                     item_cf_embeds_augmented=self.features,
                                     user_interact_matrix=self.data.aggregation('mean').tocsr(),
                                     aug_user_interact_matrix=self.data.aggregation('mean').tocsr())
        torch.set_num_threads(2)
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
            self.reference = AlphaFree(args, data)
        self.state = self.reference.state_dict()
        self.checkpoint = self.root / 'data/pretrained/amazon_movie.pth.tar'
        self.checkpoint.parent.mkdir()
        torch.save(dict(epoch=4, state_dict=self.state), self.checkpoint)
        info = self.root / 'data/amazon_movie/item_info'
        info.mkdir()
        np.save(info / 'item_cf_embeds_large3_array.npy', self.features)

    def tearDown(self):
        self.tmp.cleanup()

    def make_model(self, device='cpu'):
        return AlphaFree3(self.features, load_frozen_state(self.checkpoint), self.data.train,
                          self.data.n_users, device, encode_batch_size=2)

    def argv(self, name='search'):
        return ['--dataset', 'Movie', '--data-root', str(self.root / 'data'), '--device', 'cpu',
                '--threads', '2', '--output', str(self.root / name)]

    def test_mean_matches_original_checkpoint_scores(self):
        model = self.make_model()
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
            expected = self.reference.predict([0, 1, 2, 3])
        np.testing.assert_allclose(model.score_all([0, 1, 2, 3]).numpy(), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(set(model.state_dict()), set(self.state))

    def test_all_29_change_only_B_and_parameters_stay_frozen(self):
        model = self.make_model()
        before = model.parameter_hash()
        items = model.item_vectors.clone()
        ptr = model.item_vectors.data_ptr()
        for setting in aggregation_grid():
            model.set_aggregation(setting['mode'], setting['a'], setting['b'])
            scores = model.score_all([0, 1, 2, 3])
            expected_profile = torch.tensor(model.aggregation.toarray()) @ model.features
            expected = F.normalize(model.mlp_origin(expected_profile), dim=-1) @ items.T
            torch.testing.assert_close(scores, expected)
            self.assertFalse(scores.requires_grad)
            self.assertTrue(torch.isfinite(scores).all())
            self.assertEqual(model.item_vectors.data_ptr(), ptr)
        self.assertEqual(model.parameter_hash(), before)
        torch.testing.assert_close(model.item_vectors, items)
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in model.parameters()))
        with self.assertRaisesRegex(RuntimeError, 'frozen inference'):
            model.train()
        with self.assertRaisesRegex(RuntimeError, 'no training loss'):
            model(None)

    def test_strict_loading_and_feature_dimension(self):
        state = dict(self.state)
        del state['mlp_aug.0.bias']
        with self.assertRaises(RuntimeError):
            AlphaFree3(self.features, state, self.data.train, 4)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            AlphaFree3(np.zeros((5, 9)), self.state, self.data.train, 4)
        path = self.root / 'plain.pt'
        torch.save(self.state, path)
        self.assertEqual(set(load_frozen_state(path)), set(self.state))

    def test_complete_search_valid_only_then_selected_test_and_resume(self):
        with redirect_stdout(io.StringIO()), patch.object(runner, 'evaluate', wraps=evaluate) as evaluator, \
                patch('torch.optim.Adam', side_effect=AssertionError('No optimizer allowed')):
            report = runner.main(self.argv() + ['--search'])
        self.assertTrue(report['complete'])
        self.assertEqual(report['completed'], 29)
        self.assertEqual([call.args[2] for call in evaluator.call_args_list], ['valid'] * 29)
        selected = self.root / 'search/selection.json'
        self.assertTrue(selected.is_file())
        with redirect_stdout(io.StringIO()), patch.object(runner, 'evaluate', side_effect=AssertionError('Must resume')):
            self.assertEqual(runner.main(self.argv() + ['--search'])['completed'], 29)
        with redirect_stdout(io.StringIO()), patch.object(runner, 'evaluate', wraps=evaluate) as evaluator:
            result = runner.main(self.argv('test') + ['--split', 'test', '--selection', str(selected)])
        self.assertEqual(result['completed'], 1)
        self.assertEqual([call.args[2] for call in evaluator.call_args_list], ['test'])
        self.assertIsNone(result['best_validation'])

    def test_smoke_partial_and_selection_rejection(self):
        with redirect_stdout(io.StringIO()):
            partial = runner.main(self.argv() + ['--search', '--max-settings', '1'])
            smoke = runner.main(self.argv('smoke') + ['--search', '--eval-users', '1'])
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['results'][0]['setting']['id'], 'mean')
        self.assertFalse((self.root / 'search/selection.json').exists())
        self.assertIsNone(smoke['best_validation'])
        self.assertFalse((self.root / 'smoke/selection.json').exists())
        fake = self.root / 'fake_selection.json'
        fake.write_text(json.dumps({'model': 'AlphaFree3', 'complete': True, 'smoke_test': True}), encoding='utf-8')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'non-smoke'):
            runner.main(self.argv('reject') + ['--split', 'test', '--selection', str(fake)])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Manifest changed'):
            runner.main(self.argv() + ['--search', '--eval-users', '1'])

    def test_cannot_search_test_and_lock_preserved(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(['--search', '--split', 'test'])
        folder = self.root / 'search'
        folder.mkdir()
        (folder / 'running.lock').write_text('existing worker', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            runner.main(self.argv())
        self.assertEqual((folder / 'running.lock').read_text(), 'existing worker')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_cpu_parity(self):
        cpu = self.make_model()
        gpu = self.make_model('cuda:0')
        for setting in aggregation_grid():
            for model in (cpu, gpu):
                model.set_aggregation(setting['mode'], setting['a'], setting['b'])
            torch.testing.assert_close(cpu.score_all([0, 1, 2, 3]), gpu.score_all([0, 1, 2, 3]).cpu(), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
