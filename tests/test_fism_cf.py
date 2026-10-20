import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from cf.data import CFData
from cf.evaluate import evaluate
from cf.models import build_model, model_variants
from search_cf import build_jobs, read_json
from train_cf import ROOT, load_checkpoint, parse_args, run_one


class FISMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, content in dict(train='0 0 1\n1 2\n2\n', valid='0 2\n1 3\n2 0\n',
                                  test='0 3\n1 0\n2 1\n').items():
            (self.root / (name + '.txt')).write_text(content, encoding='utf-8')
        self.data = CFData(self.root)
        self.args = parse_args(['--model', 'fism', '--dim', '4', '--fism-alpha', '.5',
                                '--epochs', '1', '--batch-size', '2', '--max-steps', '1',
                                '--validation-only', '--output-root', str(self.root / 'runs')])
        torch.set_num_threads(2)
        self.model = build_model(self.data, self.args, 'fism', 'native', 'cpu')

    def tearDown(self):
        self.tmp.cleanup()

    def reference(self):
        name = '_fism_reference_tests'
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT.parent / 'MixGCF/modules')]
        sys.modules[name] = package
        cls = importlib.import_module(name + '.FISM').FISM
        args = types.SimpleNamespace(dim=4, alpha=self.model.alpha, user_bias=True,
                                     exclude_target=True, reg_weights=[self.args.fism_reg] * 2)
        ref = cls(dict(n_users=3, n_items=4), args, self.data.train)
        with torch.no_grad():
            ref.item_src_embedding.weight[:4].copy_(self.model.src.weight)
            ref.item_dst_embedding.weight.copy_(self.model.dst.weight)
            ref.user_bias.copy_(self.model.user_bias)
            ref.item_bias.copy_(self.model.item_bias)
        return ref

    def test_scores_match_local_reference_and_full_catalog(self):
        users = torch.arange(3).repeat_interleave(4)
        items = torch.arange(4).repeat(3)
        with torch.no_grad():
            self.model.user_bias.copy_(torch.tensor([.2, .1, -.3]))
            self.model.item_bias.copy_(torch.arange(4) * .1)
        for alpha in (0., .5, 1.):
            self.model.alpha = alpha
            expected = self.reference().score_pairs(users, items)
            torch.testing.assert_close(self.model.score_pairs(users, items), expected)
            torch.testing.assert_close(self.model.score_all([0, 1, 2]).flatten(), expected)
            self.assertTrue(torch.isfinite(self.model.score_all([2])).all())

    def test_objective_and_all_gradients_match_reference(self):
        ref = self.reference()
        batch = self.data.batch(self.data.pairs, self.model, 'cpu', 5)
        self.assertEqual(tuple(batch['neg_items'].shape), (3, 4))
        for user, negatives in zip(batch['users'].tolist(), batch['neg_items'].tolist()):
            self.assertFalse(set(negatives) & self.data.train_sets[user])
        targets = torch.cat([batch['pos_items'][:, None], batch['neg_items']], 1)
        labels = torch.zeros_like(targets, dtype=torch.float32)
        labels[:, 0] = 1
        ref_batch = dict(users=batch['users'][:, None].expand_as(targets).reshape(-1),
                         items=targets.reshape(-1), labels=labels.reshape(-1))
        losses, expected = self.model(batch), ref(ref_batch)
        for a, b in zip(losses, expected):
            torch.testing.assert_close(a, b)
        losses[0].backward()
        expected[0].backward()
        for a, b in [(self.model.src.weight.grad, ref.item_src_embedding.weight.grad[:4]),
                     (self.model.dst.weight.grad, ref.item_dst_embedding.weight.grad),
                     (self.model.user_bias.grad, ref.user_bias.grad),
                     (self.model.item_bias.grad, ref.item_bias.grad)]:
            torch.testing.assert_close(a, b)

    def test_singleton_exclusion_and_item_zero_is_real(self):
        self.model.alpha = 0.
        self.model.reg_weight = 0.
        users, items = torch.tensor([1]), torch.tensor([2])
        self.assertEqual(float(self.model.score_pairs(users, items)), 0.)
        self.model.score_pairs(users, items).sum().backward()
        self.assertEqual(float(self.model.src.weight.grad.abs().sum()), 0.)
        self.model.zero_grad()
        self.model.score_pairs(torch.tensor([0]), torch.tensor([3])).sum().backward()
        self.assertGreater(float(self.model.src.weight.grad[0].abs().sum()), 0.)

    def test_native_has_no_graph_dependency_or_user_vector_table(self):
        self.args.fism_user_bias = 0
        self.args.itemgcn_root = self.root / 'absent'
        with patch.object(self.data, 'graph', side_effect=AssertionError('No GNN')):
            model = build_model(self.data, self.args, 'fism', 'native', 'cpu')
        self.assertEqual(set(dict(model.named_parameters())), {'src.weight', 'dst.weight', 'item_bias'})
        self.assertEqual(set(model.state_dict()), {'src.weight', 'dst.weight', 'item_bias'})
        self.assertTrue(torch.isfinite(model.score_all([2])).all())

    def test_checkpoint_runner_and_masked_evaluation(self):
        result = run_one(self.data, self.args, 'fism', 'native', torch.device('cpu'), None, None)
        self.assertIsNone(result['test'])
        saved = load_checkpoint(Path(result['run_dir']) / 'best.pt', 'cpu')
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.assertEqual(result['valid'], evaluate(self.model, self.data))
        self.assertEqual(evaluate(self.model, self.data, batch_size=1), evaluate(self.model, self.data))
        self.assertEqual(saved['metadata']['source_sha256'], {})

    def test_search_native_grid_and_invalid_variants(self):
        spec = read_json(ROOT / 'configs/cf_joint_search.json')
        for phase in ('both', 'native'):
            jobs = build_jobs(spec, ['amazon_movie'], ['fism'], phase, [2025])
            self.assertEqual(len(jobs), 9)
            self.assertEqual({j['variant'] for j in jobs}, {'native'})
            self.assertEqual({j['B'] for j in jobs}, {'none'})
        self.assertEqual(model_variants('fism', 'item-only'), [])
        with self.assertRaises(SystemExit):
            parse_args(['--model', 'fism', '--variant', 'item-only'])
        with self.assertRaises(ValueError):
            build_model(self.data, self.args, 'fism', 'original', 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_fism(self):
        model = self.model.to('cuda:0')
        batch = self.data.batch(self.data.pairs, model, 'cuda:0', 5)
        loss = model(batch)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        evaluate(model, self.data)


if __name__ == '__main__':
    unittest.main()
