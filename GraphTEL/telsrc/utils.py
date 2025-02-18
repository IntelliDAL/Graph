import argparse
import random
import numpy as np
import dgl
import torch

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def contr_cos_loss(x, x_aug, T):
    batch_size = len(x.t())
    x = torch.nn.functional.normalize(x)
    x_aug = torch.nn.functional.normalize(x_aug)
    sim_matrix = x.t().mm(x_aug)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def co_data(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def def_args():
    parser = argparse.ArgumentParser(description="TEL")
    parser.add_argument("--seeds", type=int, default=[0])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_hidden", type=int, default=256)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--use5", type=bool, default=False)
    parser.add_argument("--use6", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--linear_prob", action="store_true", default=False)
    parser.add_argument("--scheduler", type=bool, default=False)
    parser.add_argument("--concat_hidden", type=bool, default=False)
    parser.add_argument("--deg4feat", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args



