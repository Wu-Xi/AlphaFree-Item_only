"""Train local ItemGCN baselines on AlphaFree data without language-model dependencies."""
import argparse
import datetime
import inspect
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from cf.data import CFData, DATASETS, file_hash
from cf.evaluate import evaluate
from cf.models import MODEL_FAMILIES, build_model, model_variants, source_hashes


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='Movie', help='Movie/Book/Video/Baby/Steam/Beauty/Health or canonical name')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--duplicate-items', choices=['error', 'count'], default='error',
                   help='count preserves repeated interactions in sampling, history weights and degrees')
    p.add_argument('--itemgcn-root', type=Path, default=ROOT.parent / 'ItemGCN')
    p.add_argument('--output-root', type=Path, default=ROOT / 'weights' / 'cf')
    p.add_argument('--model', choices=list(MODEL_FAMILIES) + ['rns', 'all'], default='stabcf')
    p.add_argument('--variant', choices=['original', 'item-only', 'both', 'native'], default='both')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--patience', type=int, default=20, help='Validation evaluations without improvement')
    p.add_argument('--validation-only', action='store_true', help='Tuning: do not compute test metrics')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--context-hops', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--l2', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--pool', choices=['mean', 'sum', 'concat', 'final'], default='mean')
    p.add_argument('--no-embedding0', dest='embedding0', action='store_false', default=True)
    p.add_argument('--n-negs', type=int, default=64, help='StabCF candidate negatives; RNS/SimGCL always use one')
    p.add_argument('--window-length', type=int, default=5)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--eps', type=float, default=.1)
    p.add_argument('--cl-rate', type=float, default=.5)
    p.add_argument('--ssl-temp', type=float, default=.2)
    p.add_argument('--encoder', choices=['mf', 'lightgcn'], default='lightgcn', help='DirectAU encoder')
    p.add_argument('--gamma', type=float, default=1., help='DirectAU uniformity weight')
    p.add_argument('--fism-alpha', type=float, default=.5, help='Original FISM history exponent; item-only uses B instead')
    p.add_argument('--fism-reg', type=float, default=.001, help='FISM unsquared global norm coefficient for each item table')
    p.add_argument('--fism-negatives', type=int, default=4, help='FISM BCE negatives per positive')
    p.add_argument('--fism-user-bias', type=int, choices=[0, 1], default=1, help='Scalar user bias, not a user embedding')
    p.add_argument('--b-mode', choices=['sum', 'mean', 'sqrt', 'sym', 'degree', 'weighted_mean'], default='sym')
    p.add_argument('--b-a', type=float, default=.5)
    p.add_argument('--b-b', type=float, default=.5)
    p.add_argument('--max-steps', type=int, default=0, help='SMOKE ONLY: cap batches per epoch; 0 means full')
    p.add_argument('--eval-users', type=int, default=0, help='SMOKE ONLY: evaluate first N users; 0 means all')
    p.add_argument('--log-every', type=int, default=50)
    args = p.parse_args(argv)
    if args.model == 'rns':
        args.model = 'lightgcn-rns'
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'eval_every', 'patience', 'top_k',
                'threads', 'dim', 'n_negs', 'window_length', 'log_every', 'fism_negatives'):
        if getattr(args, key) < 1:
            p.error(key + ' must be positive')
    if args.context_hops < 0 or min(args.max_steps, args.eval_users) < 0:
        p.error('hops and smoke limits must be nonnegative')
    if args.context_hops == 0 and (args.model in ('all', 'simgcl') or not args.embedding0):
        p.error('SimGCL or excluding layer 0 requires at least one propagation layer')
    for key in ('lr', 'alpha', 'ssl_temp'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(key + ' must be finite and positive')
    for key in ('l2', 'weight_decay', 'eps', 'cl_rate', 'gamma', 'b_a', 'b_b', 'fism_alpha', 'fism_reg'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(key + ' must be finite and nonnegative')
    args.dataset = DATASETS.get(args.dataset.lower(), args.dataset)
    if args.dataset not in DATASETS.values():
        p.error('Unknown dataset: ' + args.dataset)
    if args.model != 'all' and not model_variants(args.model, args.variant):
        p.error('Use original/item-only/both; native is a legacy alias for FISM original')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str, allow_nan=False)


def load_checkpoint(path, device):
    options = dict(map_location=device)
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    return torch.load(path, **options)


def run_one(data, args, family, variant, device, graph, aggregation, run_dir=None):
    seed_everything(args.seed)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = (Path(run_dir) if run_dir is not None else
               args.output_root / args.dataset / (family + '_' + variant) / (stamp + '_seed' + str(args.seed)))
    run_dir.mkdir(parents=True, exist_ok=False)
    smoke = bool(args.max_steps or args.eval_users)
    model = build_model(data, args, family, variant, device, graph, aggregation)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model=family, variant=variant, device=str(device))
    metadata = dict(config=config, data=data.stats, smoke_test=smoke,
                    protocol='AlphaFree full catalog: valid masks train+test; test masks train+valid; target retained',
                    sampling='shuffled interaction pairs; negatives exclude training positives only',
                    negative_sample_count=getattr(model, 'negative_sample_count', 0),
                    requires_history=getattr(model, 'requires_history', False),
                    aggregation_protocol=getattr(model, 'aggregation_protocol', None),
                    evaluation_truth='unique item IDs, matching AlphaFree C++ evaluator',
                    parameters=sum(p.numel() for p in model.parameters()),
                    parameter_shapes={n: list(p.shape) for n, p in model.named_parameters()},
                    source_sha256={} if family == 'fism' else source_hashes(args.itemgcn_root), torch_version=str(torch.__version__),
                    adapter_sha256={str(p.relative_to(ROOT)): file_hash(p)
                                    for p in [Path(__file__).resolve()] + sorted((ROOT / 'cf').glob('*.py'))})
    write_json(run_dir / 'config.json', metadata)
    print('\n{} / {} | {} parameters | {}'.format(family, variant, metadata['parameters'], run_dir), flush=True)
    best, best_epoch, stale = -float('inf'), 0, 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = np.random.permutation(len(data.pairs))
        totals, steps = np.zeros(3), 0
        epoch_start = time.perf_counter()
        for offset in range(0, len(indices), args.batch_size):
            if args.max_steps and steps >= args.max_steps:
                break
            pairs = data.pairs[indices[offset:offset + args.batch_size]]
            batch = data.batch(pairs, model, device, args.window_length)
            optimizer.zero_grad()
            losses = model(batch, epoch - 1)
            if not all(torch.isfinite(x).all() for x in losses):
                raise FloatingPointError('Nonfinite loss at epoch {}, step {}'.format(epoch, steps))
            losses[0].backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError('Nonfinite gradient: ' + name)
            optimizer.step()
            totals += [float(x.detach()) for x in losses]
            steps += 1
            if steps % args.log_every == 0:
                print('epoch {} step {} loss {:.6f}'.format(epoch, steps, totals[0] / steps), flush=True)
        record = dict(epoch=epoch, steps=steps, losses=(totals / steps).tolist(),
                      train_seconds=time.perf_counter() - epoch_start)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid = evaluate(model, data, 'valid', args.top_k, args.eval_batch_size, args.eval_users)
            record['valid'] = valid
            score = valid['recall@{}'.format(args.top_k)]
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, metadata=metadata,
                                valid=valid), run_dir / 'best.pt')
            else:
                stale += 1
        with (run_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        if stale >= args.patience:
            break
    checkpoint = load_checkpoint(run_dir / 'best.pt', device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # Test is never used for checkpoint selection. It is evaluated only after selection.
    result = dict(model=family, variant=variant, dataset=args.dataset, seed=args.seed,
                  smoke_test=smoke, best_epoch=best_epoch, parameters=metadata['parameters'],
                  valid=checkpoint['valid'],
                  test=None if args.validation_only else evaluate(
                      model, data, 'test', args.top_k, args.eval_batch_size, args.eval_users),
                  validation_only=args.validation_only,
                  elapsed_seconds=time.perf_counter() - start_time, run_dir=str(run_dir))
    write_json(run_dir / 'result.json', result)
    print('RESULT ' + json.dumps(result, allow_nan=False), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available in this Python environment')
    data = CFData(args.data_root / args.dataset / 'cf_data', args.duplicate_items)
    print(json.dumps(data.stats), flush=True)
    if args.max_steps or args.eval_users:
        print('SMOKE TEST: truncated training/evaluation; not a benchmark result.', flush=True)
    families = MODEL_FAMILIES if args.model == 'all' else [args.model]
    runs = [(family, variant) for family in families for variant in model_variants(family, args.variant)]
    graph = data.graph() if any(family != 'fism' for family, _ in runs) else None
    aggregation = (data.aggregation(args.b_mode, args.b_a, args.b_b)
                   if any(variant == 'item-only' and family != 'fism' for family, variant in runs) else None)
    results = [run_one(data, args, family, variant, device, graph, aggregation)
               for family, variant in runs]
    summary = args.output_root / args.dataset / ('summary_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    write_json(summary, results)
    print('Suite summary: ' + str(summary), flush=True)
    return results


if __name__ == '__main__':
    main()
