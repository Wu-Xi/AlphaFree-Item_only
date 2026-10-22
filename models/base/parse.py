import argparse
import os
import yaml
import math


def build_parser():
    parser = argparse.ArgumentParser()

    # Phase Args
    parser.add_argument('--phase', type=str, default='train',
                        choices=['preprocessing', 'train', 'inference'],
                        help='The phase to run the model in.')
    parser.add_argument('--dataset', type=str, default='amazon_video',
                        help='Dataset name.')
    # General Args
    parser.add_argument('--model_name', type=str, default='AlphaFree',
                        help='model name.')    
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--test_only', action="store_true",
                        help='Whether to test only.')
    parser.add_argument('--clear_checkpoints', action="store_true", default=True,
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument('--saveID', type=str, default='Saved',
                        help='Specify model save path. Description of the experiment')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--verbose', type=float, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping point.')

    # Model Args
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for optimizer.')

    
    parser.add_argument('--Ks', type = int, default= 20,
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--neg_sample',type=int,default=256)
    parser.add_argument('--infonce', type=int, default=1,
                help='whether to use infonce loss or not')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--max2keep', type=int, default=1,
                        help='max checkpoints to keep')
    
    parser.add_argument('--tau_r', type=float, default=0.1,
                    help='tau_r')
    parser.add_argument('--lm_model', type=str, default='llama',
                choices=['v3', 'llama'],
                help='The base language model')
    parser.add_argument('--tau_a', type=float, default=0,
                help='tau_a')
    parser.add_argument('--K_c', type=int, default=0,
                help='K_c')
    parser.add_argument('--lambda_align', type=float, default=0.07,
                        help='align cl loss temperature')

    # AlphaFree2 only: ignored by the original AlphaFree model.
    parser.add_argument('--b-mode', '--b_mode', dest='b_mode',
                        choices=['sum', 'mean', 'sqrt', 'sym', 'degree', 'weighted_mean'], default='mean')
    parser.add_argument('--b-a', '--b_a', dest='b_a', type=float, default=.5)
    parser.add_argument('--b-b', '--b_b', dest='b_b', type=float, default=.5)
    parser.add_argument('--validation-only', dest='validation_only', action='store_true',
                        help='AlphaFree/AlphaFree2: shared validation-only training; no test recommendations')
    parser.add_argument('--final-test', action='store_true',
                        help='With validation-only training, evaluate best checkpoint on test once at the end')

    return parser

def load_dataset_config(config_dir: str, dataset: str):
    path = os.path.join(config_dir, f"{dataset}.yaml")
    if not os.path.exists(path):
        return {}, path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict at top-level: {path}")
    return cfg, path


def parse_args(argv=None):
    # 1) minimal parse to know dataset/config_dir
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument('--dataset', type=str, default='amazon_video')
    mini.add_argument('--config_dir', type=str, default='./configs')
    mini_args, _ = mini.parse_known_args(argv)

    # 2) load config by dataset
    cfg, cfg_path = load_dataset_config(mini_args.config_dir, mini_args.dataset)

    # 3) build full parser and set defaults from config
    parser = build_parser()
    parser.add_argument('--config_dir', type=str, default=mini_args.config_dir)
    parser.add_argument('--config_path', type=str, default=cfg_path, help='Loaded config path (auto).')

    # config에 있는 key만 parser에 존재하는 옵션이면 반영
    valid_keys = {a.dest for a in parser._actions}
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    parser.set_defaults(**filtered)
    parser.set_defaults(dataset=mini_args.dataset)

    # 4) final parse (CLI가 config 덮어씀)
    args = parser.parse_args(argv)
    if args.final_test and not args.validation_only:
        parser.error('--final-test requires --validation-only')
    if args.model_name == 'AlphaFree2' or args.validation_only:
        if args.model_name not in ('AlphaFree', 'AlphaFree2'):
            parser.error('validation-only training supports AlphaFree and AlphaFree2')
        if args.phase != 'train' or args.test_only:
            parser.error('AlphaFree2 currently supports fresh training only; author pretrained weights are not AlphaFree2 weights')
        if args.infonce != 1 or args.neg_sample < 1:
            parser.error('AlphaFree2 requires InfoNCE with explicit positive-count negative sampling')
        for key in ('b_a', 'b_b', 'lambda_align'):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
                parser.error(key + ' must be finite and nonnegative')
        for key in ('lr', 'tau_a', 'tau_r'):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                parser.error(key + ' must be finite and positive')
        if args.verbose < 1 or not float(args.verbose).is_integer() or args.max_epoch < args.verbose:
            parser.error('AlphaFree2 needs an integer evaluation interval and at least one evaluation')
        if min(args.batch_size, args.hidden_size, args.patience, args.Ks, args.max2keep) < 1 or args.num_workers < 0:
            parser.error('Invalid AlphaFree2 training sizes')

    # special args(옵션 목록) 필요하면 그대로 유지 가능
    special_args = sorted(list(filtered.keys()))
    if args.model_name == 'AlphaFree2':
        special_args = sorted(set(special_args) | {'b_mode', 'b_a', 'b_b'})
    return args, special_args
