import numpy as np
from dgl.nn.pytorch.glob import AvgPooling
from dgl.dataloading import GraphDataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from telsrc.utils import def_args, set_random_seed, contr_cos_loss, co_data
from telsrc.data_util import load_graph_dataset
from telsrc.models.edcoder import define_tel
from telsrc.evaluation import graph_classification_evaluation

def pretrain(model, dataloaders, optimizer, max_epoch, device):
    train_loader, _ = dataloaders
    for epoch in range(max_epoch):
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g, _ = batch
            batch_g = batch_g.to(device)
            recPR_loss, (nembr, gembr), (nembw, gembw) = model(batch_g, batch_g.ndata["attr"])
            graph_contr_loss = contr_cos_loss(gembr, gembw, T=0.14)
            node_contr_loss = 0
            for i in range(0, len(nembr)):
                node_contr_loss = node_contr_loss + contr_cos_loss(nembr[i], nembw[i], T=0.14)
            loss = recPR_loss + (1-recPR_loss)*(node_contr_loss + graph_contr_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(f"Epoch {epoch}|train_loss:{np.mean(loss_list):.4f}")
    return model

def main(args):
    graphs, (num_features, num_classes) = load_graph_dataset(args.dataset, deg4feat=args.deg4feat)
    args.num_features = num_features
    args.fea_len = graphs[0][0].ndata["attr"].shape[1]
    args.num_classes = num_classes
    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, co_data=co_data, batch_size=args.batch_size, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, co_data=co_data, batch_size=args.batch_size, shuffle=False)
    acc_list = []
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        model = define_tel(args)
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
        model = pretrain(model, (train_loader, eval_loader), optimizer, args.max_epoch, args.device)
        model.eval()
        test_acc = graph_classification_evaluation(model, AvgPooling(), eval_loader, num_classes, args.lr_f, args.weight_decay_f, args.max_epoch_f, args.device, mute=False)
        acc_list.append(test_acc)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")

if __name__ == "__main__":
    args = def_args()
    main(args)

