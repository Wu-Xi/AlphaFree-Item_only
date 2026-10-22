"""Shared validation-selected trainer, with optional one-time final test; model/loss unchanged."""
import json
import os
import time
import math
from pathlib import Path

from .RS import AlphaFreeRS
from .utils import evaluation


class AlphaFree2RS(AlphaFreeRS):
    def train(self):
        # Same optimizer/epoch/evaluation loop as the parent. Its final plotting
        # helper requires equal valid/test curves (and >=2 evals), so omit plotting
        # here rather than evaluating test during hyperparameter selection.
        self.set_optimizer()
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.flag:
                break
            started = time.time()
            losses = self.train_one_epoch(epoch)
            if not all(math.isfinite(value) for value in losses):
                raise FloatingPointError('Nonfinite AlphaFree2 training loss')
            self.document_running_loss(losses, epoch, time.time() - started)
            if (epoch + 1) % self.verbose == 0:
                self.eval_and_check_early_stop(epoch)

    def preperation_for_saving(self, args, special_args):
        super().preperation_for_saving(args, special_args)
        # The original trainer sometimes concatenates filenames without a slash.
        self.base_path = self.base_path.rstrip('/\\') + os.sep

    def get_evaluators(self, data):
        evaluators, names = super().get_evaluators(data)
        if self.args.validation_only:
            return [evaluators[0]], [names[0]]
        return evaluators, names

    def recommend_top_k(self):
        if not self.args.validation_only:
            super().recommend_top_k()

    def execute(self):
        super().execute()
        # Parent execute has restored the validation-best checkpoint. Test does
        # not participate in early stopping, checkpointing, or parameter selection.
        test_metrics = None
        if self.args.final_test:
            evaluators, names = super().get_evaluators(self.data)
            test_evaluator = evaluators[names.index('test')]
            label = 'Final test@{} at validation-best epoch={}'.format(
                self.args.Ks, self.data.best_valid_epoch)
            print(label, flush=True)
            with (Path(self.base_path) / 'stats.txt').open('a', encoding='utf-8') as stream:
                stream.write(label + '\n')
            _, _, metrics = evaluation(self.args, self.data, self.model,
                self.data.best_valid_epoch, self.base_path, test_evaluator, 'test')
            test_metrics = {key: float(value) for key, value in metrics.items()}
            if not all(math.isfinite(value) for value in test_metrics.values()):
                raise FloatingPointError('Nonfinite final test metrics; checkpoint retained')
        result = dict(model=self.args.model_name, dataset=self.args.dataset, seed=self.args.seed,
                      aggregation_protocol='count-weighted train history; original user branch only',
                      aggregation_scope='original-user-only', b_mode=self.args.b_mode,
                      b_a=self.args.b_a, b_b=self.args.b_b,
                      validation_only=self.args.validation_only,
                      final_test=self.args.final_test, test_k=self.args.Ks,
                      test_metrics=test_metrics,
                      best_valid_recall=float(self.data.best_valid_recall),
                      best_epoch=int(self.data.best_valid_epoch),
                      checkpoint=str((Path(self.base_path) /
                                      ('epoch={}.checkpoint.pth.tar'.format(self.data.best_valid_epoch))).resolve()))
        with (Path(self.base_path) / 'result.json').open('w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        print('ALPHAFREE2_RESULT ' + json.dumps(result, allow_nan=False), flush=True)
