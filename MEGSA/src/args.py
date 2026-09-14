"""Purpose: Parse MEGSA runtime arguments and dataset settings.

Notes: B denotes batch size, N node count, M motif count, D embedding dimension, and F node feature dimension.
"""

import argparse
from pathlib import Path

from tabulate import tabulate

from src.runtime import PROJECT_ROOT


def str2bool(str):
    """Purpose: Check whether the lowercased string equals true.

    Args:
        str: String to evaluate.

    Returns:
        Boolean result; any other string returns False.
    """
    return True if str.lower() == 'true' else False


def device_argument(value):
    """Validate a CPU device, CUDA device string, or nonnegative GPU index.

    Args:
        value: Device argument from the command line.

    Returns:
        A validated device string.
    """
    if value == 'cpu':
        return value
    if not value.removeprefix('cuda:').isdecimal():
        raise argparse.ArgumentTypeError('Use cpu, a GPU index such as 0, or cuda:0.')
    return value


def get_args(argv=None):
    """Purpose: Parse command-line arguments, attach dataset settings, and print the configuration table.

    Args:
        argv: Optional argument list; defaults to the process command line.

    Returns:
        Namespace containing runtime arguments and dataset settings.
    """
    parser = argparse.ArgumentParser(description='Train or evaluate MEGSA.')
    parser.add_argument('--mode', choices=['pretrain', 'train', 'eval', 'all'], default='all')
    parser.add_argument('--dataset', default='AIDS700nef', choices=['AIDS700nef', 'LINUX', 'IMDBMulti'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--commit', default='', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default='0', type=device_argument)
    parser.add_argument('--data_dir', type=Path, default=PROJECT_ROOT / 'datasets')
    parser.add_argument('--model_dir', type=Path, default=PROJECT_ROOT / 'model')
    parser.add_argument('--pretrain_dir', type=Path, default=PROJECT_ROOT / 'pretrain_model')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--attention_layers', default=2, type=int)

    # Define node capacities, feature dimensions, motif counts, and loss weights for each dataset.
    dataset_info = {
        'AIDS700nef': {'max_nodes': 10, 'node_feat_dim': 29, 'motif_num': 5, 'alpha': 30},
        'LINUX': {'max_nodes': 10, 'node_feat_dim': 8, 'motif_num': 5, 'alpha': 100},
        'IMDBMulti': {'max_nodes': 89, 'node_feat_dim': 89, 'motif_num': 10, 'alpha': 200},
    }

    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.pretrain_epochs <= 0 or args.epochs <= 0:
        parser.error('Batch size and epoch counts must be positive.')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error('Use lr > 0, weight_decay >= 0, and 0 <= dropout < 1.')
    if args.attention_layers < 0:
        parser.error('attention_layers must be nonnegative.')
    for name in ('data_dir', 'model_dir', 'pretrain_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.max_nodes = dataset_info[args.dataset]['max_nodes']
    args.node_feat_dim = dataset_info[args.dataset]['node_feat_dim']
    args.motif_num = dataset_info[args.dataset]['motif_num']
    args.alpha = dataset_info[args.dataset]['alpha']

    args_table = [
        ['mode', args.mode],
        ['dataset', args.dataset],
        ['batch_size', args.batch_size],
        ['pretrain_epochs', args.pretrain_epochs],
        ['epochs', args.epochs],
        ['device', args.device],
        ['lr', args.lr],
        ['dropout', args.dropout],
        ['max_nodes', args.max_nodes],
        ['node_feat_dim', args.node_feat_dim],
        ['motif_num', args.motif_num],
    ]
    print(tabulate(args_table, headers=['Parameter', 'Value'], tablefmt='grid'))
    return args
