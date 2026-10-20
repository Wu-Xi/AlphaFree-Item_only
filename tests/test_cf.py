import importlib.util
import itertools
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from cf.data import CFData
from cf.evaluate import evaluate, excluded_items, rank_metrics
from cf.models import PAIRED_FAMILIES as MODEL_FAMILIES, build_model
from train_cf import ROOT, load_checkpoint, parse_args, run_one, seed_everything


class CFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, content in dict(train='0 0 1\n1 1 2\n2 2 3\n',
                                  valid='0 2\n1 3\n2 4\n3 0\n',
                                  test='0 3\n1 4\n2 0\n3 4\n').items():
            (self.root / (name + '.txt')).write_text(content, encoding='utf-8')
        self.data = CFData(self.root)
        self.args = parse_args(['--dim', '8', '--context-hops', '2', '--n-negs', '3',
                                '--window-length', '2', '--epochs', '1', '--batch-size', '4',
                                '--max-steps', '1', '--top-k', '2', '--eval-batch-size', '2',
                                '--output-root', str(self.root / 'runs')])
        seed_everything(42)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_only_graph_and_B(self):
        self.assertEqual((self.data.n_users, self.data.n_items), (4, 5))
        r = self.data.interactions.toarray()
        self.assertEqual(r[0, 2], 0)  # Validation edge must not enter the graph.
        self.assertEqual(r[0, 3], 0)  # Test edge must not enter the graph.
        np.testing.assert_allclose(self.data.aggregation('mean').toarray().sum(1), [1, 1, 1, 0])
        sym = self.data.aggregation().toarray()
        graph = self.data.graph().toarray()
        np.testing.assert_allclose(graph[:4, 4:], sym)
        np.testing.assert_allclose(graph, graph.T)
        np.testing.assert_allclose(graph[3], 0)
        # Compare graph construction with the existing ItemGCN implementation.
        spec = importlib.util.spec_from_file_location('reference_data', self.args.itemgcn_root / 'utils/data_loader.py')
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        reference.n_users, reference.n_items = 4, 5
        np.testing.assert_allclose(graph, reference.build_sparse_graph(self.data.pairs).toarray(), atol=1e-7)
        for mode in ('sum', 'mean', 'sqrt', 'sym', 'degree', 'weighted_mean'):
            np.testing.assert_allclose(self.data.aggregation(mode, .3, .2).toarray(),
                                       reference.build_B(self.data.pairs, mode, .3, .2).toarray(), atol=1e-7)

    def test_all_eight_gradients_sampling_checkpoint(self):
        for family in MODEL_FAMILIES:
            for variant in ('original', 'item-only'):
                with self.subTest(family=family, variant=variant):
                    model = build_model(self.data, self.args, family, variant, torch.device('cpu'))
                    names = dict(model.named_parameters())
                    self.assertEqual('user_embed' in names, variant == 'original')
                    if variant == 'item-only':
                        self.assertEqual(set(names), {'item_embed'})
                    batch = self.data.batch(self.data.pairs, model, 'cpu', 2)
                    if family == 'directau':
                        self.assertNotIn('neg_items', batch)
                    else:
                        self.assertEqual(batch['neg_items'].shape[1], 3 if family == 'stabcf' else 1)
                        for u, neg in zip(batch['users'].tolist(), batch['neg_items'].tolist()):
                            self.assertFalse(self.data.train_sets[u].intersection(neg))
                    if family == 'stabcf':
                        for u, hist in zip(batch['users'].tolist(), batch['observed_pos_items'].tolist()):
                            self.assertTrue(set(hist).issubset(self.data.train_sets[u]))
                    else:
                        self.assertNotIn('observed_pos_items', batch)
                    losses = model(batch, 0)
                    self.assertTrue(all(torch.isfinite(x) for x in losses))
                    losses[0].backward()
                    self.assertTrue(torch.isfinite(model.item_embed.grad).all())
                    self.assertGreater(float(model.item_embed.grad.abs().sum()), 0)
                    torch.optim.Adam(model.parameters()).step()
                    before = evaluate(model, self.data, k=2, batch_size=2)
                    path = self.root / 'checkpoint.pt'
                    torch.save(dict(state_dict=model.state_dict()), path)
                    restored = build_model(self.data, self.args, family, variant, torch.device('cpu'))
                    restored.load_state_dict(load_checkpoint(path, 'cpu')['state_dict'], strict=True)
                    self.assertEqual(before, evaluate(restored, self.data, k=2, batch_size=1))
                    for key in model.state_dict():
                        self.assertNotIn('norm', key)  # No graph persisted in checkpoints.

    def test_masks_and_metrics_match_reference(self):
        self.assertEqual(excluded_items(self.data, 'valid', 0), {0, 1, 3})
        self.assertEqual(excluded_items(self.data, 'test', 0), {0, 1, 2})
        self.data.splits['valid'][0].append(1)
        self.assertNotIn(1, excluded_items(self.data, 'valid', 0))
        path = ROOT / 'models/base/evaluator/backend/python/metric.py'
        spec = importlib.util.spec_from_file_location('reference_metric', path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        rank, truth = [3, 1, 0, 4], [1, 4]
        got = rank_metrics(rank, truth, 3)
        np.testing.assert_allclose(got[:2], [ref.recall(rank, truth)[2], ref.ndcg(rank, truth)[2]])

    def test_real_evaluator_masks_before_ranking(self):
        class Fixed:
            def eval(self):
                return self

            def generate(self):
                return torch.ones(4, 1), torch.arange(5, dtype=torch.float32)[:, None]

            def rating(self, u, i):
                return u @ i.T

        # User 0 valid candidates after masking are {2,4}; rank=[4,2].
        result = evaluate(Fixed(), self.data, 'valid', k=2, max_users=1)
        self.assertEqual(result['recall@2'], 1.)
        self.assertAlmostEqual(result['ndcg@2'], 1. / np.log2(3))

    def test_runner_and_best_checkpoint(self):
        result = run_one(self.data, self.args, 'directau', 'item-only', torch.device('cpu'),
                         self.data.graph(), self.data.aggregation())
        self.assertTrue(result['smoke_test'])
        self.assertEqual(result['best_epoch'], 1)
        self.assertTrue((Path(result['run_dir']) / 'best.pt').is_file())

    def test_item_only_user_embeddings_and_gradients(self):
        for family in MODEL_FAMILIES:
            model = build_model(self.data, self.args, family, 'item-only', torch.device('cpu'))
            b = torch.tensor(self.data.aggregation().toarray())
            torch.testing.assert_close(model.build_user_embed(), b @ model.item_embed)
            model.build_user_embed().sum().backward()
            torch.testing.assert_close(model.item_embed.grad, b.T @ torch.ones(4, 8))

    def test_directau_mf_singleton_tail(self):
        self.args.encoder = 'mf'
        for variant in ('original', 'item-only'):
            model = build_model(self.data, self.args, 'directau', variant, torch.device('cpu'))
            batch = self.data.batch(self.data.pairs[:1], model, 'cpu', 2)
            total, align, uniform = model(batch)
            self.assertTrue(torch.isfinite(total))
            self.assertEqual(float(uniform), 0.)
            total.backward()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_all_eight(self):
        device = torch.device('cuda:0')
        for family in MODEL_FAMILIES:
            for variant in ('original', 'item-only'):
                model = build_model(self.data, self.args, family, variant, device)
                batch = self.data.batch(self.data.pairs, model, device, 2)
                loss = model(batch)[0]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(buffer.device.type == 'cuda' for buffer in model.buffers()))
                evaluate(model, self.data, k=2)

    def test_rns_alias_and_single_negative(self):
        self.assertEqual(parse_args(['--model', 'rns']).model, 'lightgcn-rns')
        for variant in ('original', 'item-only'):
            # n_negs=3 in fixture must not turn RNS into hard-negative mining.
            model = build_model(self.data, self.args, 'lightgcn-rns', variant, torch.device('cpu'))
            self.assertEqual(model.ns, 'rns')
            self.assertEqual(model.negative_sample_count, 1)
            self.assertFalse(model.requires_history)

    def test_search_spaces_are_valid_cli_combinations(self):
        with (ROOT / 'configs/cf_search_space.json').open(encoding='utf-8') as stream:
            search = json.load(stream)
        self.assertEqual(set(search['stage1']), set(MODEL_FAMILIES))
        for family, stage in search['stage1'].items():
            keys = list(stage['grid'])
            combinations = list(itertools.product(*(stage['grid'][key] for key in keys)))
            self.assertEqual(len(combinations), 36)
            for values in combinations:
                options = dict(search['fixed'], **stage['fixed'])
                options.update(zip(keys, values))
                argv = ['--model', family]
                for key, value in options.items():
                    argv.append('--' + key)
                    if value is not True:
                        argv.append(str(value))
                args = parse_args(argv)
                self.assertTrue(args.validation_only)
                self.assertEqual(args.model, family)
        for family, expected in (('stabcf', 27), ('simgcl', 36)):
            grid = search['stage2'][family]
            self.assertEqual(len(list(itertools.product(*grid.values()))), expected)

    def test_validation_only_never_computes_test_metric(self):
        self.args.validation_only = True
        with patch('train_cf.evaluate', wraps=evaluate) as evaluator:
            result = run_one(self.data, self.args, 'lightgcn-rns', 'item-only', torch.device('cpu'),
                             self.data.graph(), self.data.aggregation())
        self.assertIsNone(result['test'])
        self.assertTrue(result['validation_only'])
        self.assertEqual([call.args[2] for call in evaluator.call_args_list], ['valid'])

    def test_input_validation(self):
        (self.root / 'train.txt').write_text('0 1 1\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            CFData(self.root)

    def test_missing_ids_rejected(self):
        (self.root / 'test.txt').write_text('0 9\n1 4\n2 0\n3 4\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            CFData(self.root)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
