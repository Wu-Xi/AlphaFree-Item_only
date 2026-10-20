import ast
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import time
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

from cf.data import CFData
from models.AlphaFree import AlphaFree
from models.AlphaFree2 import AlphaFree2
from models.base.parse import parse_args
from models.base.user_aggregation import aggregation_grid, build_user_aggregation
import search_alphafree2 as search


class AlphaFree2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        directory = self.root / 'data/amazon_movie/cf_data'
        directory.mkdir(parents=True)
        for name, content in dict(train='0 0 1\n1 1 2 3\n2 1\n3\n',
                                  valid='0 2\n1 4\n2 0\n3 2\n',
                                  test='0 3\n1 0\n2 4\n3 4\n').items():
            (directory / (name + '.txt')).write_text(content, encoding='utf-8')
        self.cf = CFData(directory)
        rng = np.random.RandomState(42)
        self.data = types.SimpleNamespace(
            train_user_list=self.cf.train, n_users=4, n_items=5,
            user_interact_matrix=self.cf.aggregation('mean').tocsr(),
            aug_user_interact_matrix=sparse.csr_matrix(np.array([
                [0, .5, .5, 0, 0], [0, 0, .5, .5, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)),
            item_cf_embeds_original=rng.normal(size=(5, 8)).astype(np.float32),
            item_cf_embeds_augmented=rng.normal(size=(5, 8)).astype(np.float32))
        self.args, _ = parse_args(['--model_name', 'AlphaFree2', '--dataset', 'amazon_movie',
                                   '--config_dir', str(search.ROOT / 'configs'), '--hidden_size', '4'])
        self.args.cuda = 'cpu'
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_29_matrices_match_existing_CF_and_no_leakage(self):
        grid = aggregation_grid()
        self.assertEqual(len(grid), 29)
        self.assertEqual(len({x['id'] for x in grid}), 29)
        from search_cf import representations, read_json
        existing = representations(read_json(search.ROOT / 'configs/cf_joint_search.json')['representations'])
        self.assertEqual([x['id'] for x in grid], [x['id'] for x in existing])
        for setting in grid:
            matrix = build_user_aggregation(self.cf.train, 4, 5, setting['mode'], setting['a'], setting['b'])
            expected = self.cf.aggregation(setting['mode'], setting['a'], setting['b'])
            np.testing.assert_allclose(matrix.toarray(), expected.toarray(), atol=1e-7)
            self.assertEqual(matrix[0, 2], 0.)  # Validation item is not a training edge.
            self.assertEqual(matrix[0, 3], 0.)
            self.assertEqual(matrix[3].nnz, 0)
            self.assertTrue(np.isfinite(matrix.data).all())

    def test_mean_exact_model_loss_gradients_and_predictions(self):
        # Original code is CUDA-only. This test redirects transfers on tiny CPU fixtures;
        # it verifies numerical equivalence, NOT GPU/compiled evaluator compatibility.
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
            original = AlphaFree(self.args, self.data)
            changed = AlphaFree2(self.args, self.data)
            changed.load_state_dict(original.state_dict(), strict=True)
            users, pos = torch.tensor([0, 1, 2]), torch.tensor([0, 2, 1])
            neg = torch.tensor([[2, 3], [0, 4], [0, 4]])
            for before, after in zip(original.compute_origin(users), changed.compute_origin(users)):
                torch.testing.assert_close(before, after)
            for before, after in zip(original.compute_aug(users), changed.compute_aug(users)):
                torch.testing.assert_close(before, after)
            loss1, loss2 = original(users, pos, neg), changed(users, pos, neg)
            torch.testing.assert_close(loss1, loss2)
            loss1.backward()
            loss2.backward()
            for (n1, p1), (n2, p2) in zip(original.named_parameters(), changed.named_parameters()):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1.grad, p2.grad)
            np.testing.assert_allclose(original.predict([0, 1, 2, 3]), changed.predict([0, 1, 2, 3]))

    def test_all_29_train_and_leave_augmented_branch_and_features_unchanged(self):
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **k: tensor):
            original = AlphaFree(self.args, self.data)
            users = torch.tensor([0, 1, 2])
            expected_aug = original.compute_aug(users)
            for setting in aggregation_grid():
                self.args.b_mode, self.args.b_a, self.args.b_b = setting['mode'], setting['a'], setting['b']
                model = AlphaFree2(self.args, self.data)
                model.load_state_dict(original.state_dict())
                torch.testing.assert_close(model.calculate_user_embedding(users),
                    torch.tensor(model.user_interact_matrix[users.numpy()].dot(self.data.item_cf_embeds_original)))
                for a, b in zip(model.compute_aug(users), expected_aug):
                    torch.testing.assert_close(a, b)
                self.assertIs(model.aug_user_interact_matrix, self.data.aug_user_interact_matrix)
                self.assertFalse(model.init_item_cf_embeds.requires_grad)
                self.assertTrue(all(name.startswith(('mlp_origin.', 'mlp_aug.')) for name, _ in model.named_parameters()))
                loss = model(users, torch.tensor([0, 2, 1]), torch.tensor([[2, 3], [0, 4], [0, 4]]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                torch.optim.Adam(model.parameters(), lr=.001).step()
            np.testing.assert_allclose(self.data.user_interact_matrix.toarray(), self.cf.aggregation('mean').toarray())

    def test_parse_defaults_cli_and_invalid_inference(self):
        args, special = parse_args(['--model_name', 'AlphaFree2', '--dataset', 'amazon_movie',
                                    '--config_dir', str(search.ROOT / 'configs'), '--b-mode', 'degree',
                                    '--b-a', '.25', '--b-b', '.75', '--validation-only'])
        self.assertEqual((args.b_mode, args.b_a, args.b_b), ('degree', .25, .75))
        self.assertEqual((args.K_c, args.tau_a, args.tau_r), (5, .2, .15))
        self.assertTrue({'b_mode', 'b_a', 'b_b'} <= set(special))
        for extra in (['--phase', 'inference'], ['--neg_sample', '-1'], ['--b-a', 'nan'], ['--max_epoch', '1']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--model_name', 'AlphaFree2', '--dataset', 'amazon_movie',
                            '--config_dir', str(search.ROOT / 'configs')] + extra)

    def test_dry_run_no_writes_and_all_commands_valid(self):
        with redirect_stdout(io.StringIO()):
            jobs = search.main(['--output', str(self.root / 'dry')])
        self.assertFalse((self.root / 'dry').exists())
        self.assertEqual(len(jobs), 29)
        self.assertTrue(all(job['parameters']['validation_only'] for job in jobs))

    def test_search_scheduler_status_resume_and_manifest_guard(self):
        dataset = self.root / 'data/amazon_movie'
        for relative in ('item_info/item_cf_embeds_large3_array.npy',
                         'preprocessed/user_interact_matrix.npz',
                         'preprocessed/aug_user_interact_matrix.npz',
                         'preprocessed/item_cf_embeds_augmented.npy'):
            path = dataset / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'fixture-only; training subprocess mocked')
        argv = ['--execute', '--data-root', str(self.root / 'data'), '--output', str(self.root / 'search'),
                '--max-runs', '1']
        def fake_train(command, **kwargs):
            args, _ = parse_args(command[3:])
            checkpoint = self.root / (args.saveID + '.pt')
            checkpoint.touch()
            result = dict(model='AlphaFree2', dataset=args.dataset, seed=args.seed,
                          validation_only=True, b_mode=args.b_mode, b_a=args.b_a, b_b=args.b_b,
                          best_valid_recall=.1, checkpoint=str(checkpoint))
            kwargs['stdout'].write('ALPHAFREE2_RESULT ' + json.dumps(result) + '\n')
            return types.SimpleNamespace(returncode=0)
        with redirect_stdout(io.StringIO()), patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1), patch.object(search.subprocess, 'run', side_effect=fake_train) as runner:
            self.assertEqual(search.main(argv)['completed'], 1)
            self.assertEqual(search.main(argv)['completed'], 2)
            self.assertEqual(runner.call_count, 2)
            self.assertNotEqual(runner.call_args_list[0].args[0], runner.call_args_list[1].args[0])
            with self.assertRaisesRegex(ValueError, 'NEW output'):
                search.main(argv + ['--lr', '.002'])
        self.assertFalse((self.root / 'search/running.lock').exists())

    def test_trainer_valid_only_control_without_compiled_dependencies(self):
        # Exercise the real subclass methods against a tiny fake parent; not an end-to-end trainer test.
        source = (search.ROOT / 'models/base/AlphaFree2RS.py').read_text(encoding='utf-8')
        definition = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef))
        class Parent:
            def get_evaluators(self, data):
                return ['valid_evaluator', 'test_evaluator'], ['valid', 'test']
            def recommend_top_k(self):
                self.exported = True
            def set_optimizer(self):
                self.steps, self.evals = [], []
            def train_one_epoch(self, epoch):
                self.steps.append(epoch)
                return [1.]
            def document_running_loss(self, *args):
                pass
            def eval_and_check_early_stop(self, epoch):
                self.evals.append(epoch)
                if epoch == 3:
                    self.flag = True
        namespace = {'AlphaFreeRS': Parent, 'time': time, 'math': math}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<AlphaFree2RS test>', 'exec'), namespace)
        trainer = namespace['AlphaFree2RS']()
        trainer.args = types.SimpleNamespace(validation_only=True)
        trainer.exported = False
        self.assertEqual(trainer.get_evaluators(None), (['valid_evaluator'], ['valid']))
        trainer.recommend_top_k()
        self.assertFalse(trainer.exported)
        trainer.args.validation_only = False
        self.assertEqual(len(trainer.get_evaluators(None)[0]), 2)
        trainer.recommend_top_k()
        self.assertTrue(trainer.exported)
        trainer.start_epoch, trainer.max_epoch, trainer.verbose = 0, 10, 2
        trainer.train()
        self.assertEqual(trainer.steps, [0, 1, 2, 3])
        self.assertEqual(trainer.evals, [1, 3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_model_without_transfer_patches(self):
        self.args.cuda = 0
        model = AlphaFree2(self.args, self.data).cuda(0)
        users = torch.tensor([0, 1, 2], device='cuda:0')
        loss = model(users, torch.tensor([0, 2, 1], device='cuda:0'),
                     torch.tensor([[2, 3], [0, 4], [0, 4]], device='cuda:0'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(np.isfinite(model.predict([0, 1, 2])).all())


if __name__ == '__main__':
    unittest.main()
