import argparse

def get_args():
    parser = argparse.ArgumentParser('Model')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--bs', type=int, default=16, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout probability for all dropout layers')
    parser.add_argument('--decay', type=float, default=0.1, help='decay in Adam')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=123, help='random seed for all randomized algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--checkpoint', type=str, help='which checkpoint for encoder')
    parser.add_argument('--save', type=str, help='which checkpoint for encoder')

    parser.add_argument('--head', type=int, default=2)

    args = parser.parse_args()

    return args


def get_dis_args():
    parser = argparse.ArgumentParser('Model')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout probability for all dropout layers')
    parser.add_argument('--decay', type=float, default=0.1, help='decay in Adam')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=123, help='random seed for all randomized algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--checkpoint', type=str, help='which checkpoint for encoder')
    parser.add_argument('--save', type=str, help='which checkpoint for encoder')

    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=1.0)

    args = parser.parse_args()

    return args
