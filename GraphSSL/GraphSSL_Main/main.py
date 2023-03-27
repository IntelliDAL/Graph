import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
import numpy as np
import dgl
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score

from frameworks.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    load_best_configs,
)
from frameworks.datasets.data_util import load_graph_classification_dataset
from frameworks.models import build_model


def contr_cos_loss(x, x_aug, T):
        batch_size = len(x)

        x = torch.nn.functional.normalize(x)
        x_aug = torch.nn.functional.normalize(x_aug)
        sim_matrix = x.mm(x_aug.t())

        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss = - torch.log(loss).mean()

        return loss


def pretrain(model, dataloaders, dataloaders_tr, max_epoch, device):
    
    train_loader, eval_loader = dataloaders
    train_loader_tr, eval_loader_tr = dataloaders_tr

    optimizer_1 = create_optimizer("adam", model, lr=0.0015, weight_decay=0.005)
    optimizer_2 = create_optimizer("adam", model, lr=0.0010, weight_decay=0.005)

    scheduler_2 = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
    scheduler_2 = torch.optim.lr_scheduler.LambdaLR(optimizer_2, lr_lambda=scheduler_2)

    A_epoch = 20 # 20-100
    B_epoch = 40 # 20-200
    max_epoch = A_epoch + B_epoch
    epoch_iter = tqdm(range(max_epoch))

    max_acc = 0

    for epoch in epoch_iter:

        train_loss_list = []

        # train
        model.train()
        if epoch <= A_epoch:
            for batch in train_loader:
                batch_g, _ = batch
                batch_g = batch_g.to(device)
                unbatch_g = dgl.unbatch(batch_g)

                feat = batch_g.ndata["attr"]
                model.train()
                rec_loss, adj_rec, adj_aug_rec, x_rec_con, x_init_con = model(batch_g, unbatch_g, feat, upstream=True)

                edge_contr_loss = contr_cos_loss(adj_rec, adj_aug_rec, T=0.07)
                node_contr_loss = contr_cos_loss(x_rec_con, x_init_con, T=0.14)

                loss = rec_loss + edge_contr_loss + node_contr_loss

                optimizer_1.zero_grad()
                loss.backward()
                optimizer_1.step()

                train_loss_list.append(loss.item())


        else:
            for batch in train_loader_tr:
                batch_g, labels_g = batch
                batch_g = batch_g.to(device)
                labels_gpu = labels_g.to(device)

                feat = batch_g.ndata["attr"]
                model.train()
                pred = model(batch_g, None, feat, upstream=False, tran=False)   # tran=True

                cla_loss = torch.nn.CrossEntropyLoss()(pred, labels_gpu.long())

                optimizer_2.zero_grad()
                cla_loss.backward()
                optimizer_2.step()
                train_loss_list.append(cla_loss.item())

            scheduler_2.step()

        # test
        test_loss_list = []
        batch_acc = []

        if epoch <= A_epoch:
            continue

        model.eval()
        with torch.no_grad():
            for batch in eval_loader_tr:
                batch_g, labels_g = batch
                batch_g = batch_g.to(device)
                labels_gpu = labels_g.to(device)

                feat = batch_g.ndata["attr"]
                model.train()
                preds = model(batch_g, None, feat, upstream=False, tran=False)   # tran=True
                test_loss = torch.nn.CrossEntropyLoss()(preds, labels_gpu.long())

                _, indices = torch.max(preds, dim=1)

                acc_ = accuracy_score(labels_g, indices.cpu())
                batch_acc.append(acc_)
                test_loss_list.append(test_loss.item())

        oplr = optimizer_2.state_dict()["param_groups"][0]["lr"]

        if max_acc < np.mean(batch_acc):
            max_acc = np.mean(batch_acc)
            # save_model = model

        epoch_iter.set_description(f"Epoch {epoch+1} | train_loss: {np.mean(train_loss_list):.4f} | test_loss: {np.mean(test_loss_list):.4f} | oplr: {oplr:.4f} | max_acc: {max_acc:.4f} | test_acc: {np.mean(batch_acc):.4f}")

    print("Final ACC = {:.4f}".format(np.mean(batch_acc)))

    return model

            
def collate_fn(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def main(args):
    # device = args.device if args.device >= 0 else "cpu"
    device = "cuda"
    args.device = "cuda"
    seeds = args.seeds
    max_epoch = args.max_epoch

    deg4feat = args.deg4feat
    batch_size = 32 # args.batch_size

    pretraindata = "PROTEINS" # "MUTAG" # "IMDB-BINARY" # "REDDIT-BINARY" # "COLLAB" # "IMDB-MULTI" # "DD"
    funedata = "PROTEINS" # "MUTAG" #  "IMDB-BINARY" # "REDDIT-BINARY" # "COLLAB" # "IMDB-MULTI" # "DD"

    graphs, (num_features, num_classes) = load_graph_classification_dataset(pretraindata, deg4feat=deg4feat)

    args.num_features = num_features
    args.fea_len = graphs[0][0].ndata["attr"].shape[1]

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    graphs_tr, (num_features_tr, num_classes_tr) = load_graph_classification_dataset(funedata, deg4feat=deg4feat)
    
    train_idx_tr = torch.arange(len(graphs_tr))
    train_sampler_tr = SubsetRandomSampler(train_idx_tr)
    train_loader_tr = GraphDataLoader(graphs_tr, sampler=train_sampler_tr, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    eval_loader_tr = GraphDataLoader(graphs_tr, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    args.num_classes = num_classes_tr


    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        model.to(device)

        model = pretrain(model,
                         (train_loader, eval_loader_tr), 
                         (train_loader_tr, eval_loader_tr), 
                         max_epoch, device)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)




# IMDB-BINARY T1=T2=0.14 PROTEINS
# optimizer_1 = create_optimizer("adam", model, lr=0.0015, weight_decay=0.005)
# optimizer_2 = create_optimizer("adam", model, lr=0.0010, weight_decay=0.005)
# python main.py --lr 0.00015 --lr_f 0.005 --num_hidden 512 --num_heads 2 --weight_decay 0 --weight_decay_f 0 --max_epoch 60 --max_epoch_f 500 --mask_rate 0.5 --num_layers 2 --encoder gcn --decoder gcn --activation prelu --in_drop 0.2 --loss_fn sce --optimizer adam --replace_rate 0.0 --drop_edge_rate 0.0 --alpha_l 1 --norm batchnorm --pooling mean --batch_size 32 --alpha_l 1

# MUTAG
# python main.py --num_hidden 32 --num_layers 5 --mask_rate 0.5 --encoder gcn --decoder gcn --activation prelu --batch_size 64 --alpha_l 2 --norm batchnorm     

# IMDB-MULTI: 
# python main.py --lr 0.00015 --num_hidden 512 --num_heads 2 --weight_decay 0 --max_epoch 200 --mask_rate 0.5 --num_layers 3 --encoder gcn --decoder gcn --activation prelu --in_drop 0.2 --loss_fn sce --optimizer adam --replace_rate 0.0 --drop_edge_rate 0.0 --alpha_l 1 --norm batchnorm --pooling mean --batch_size 32 --alpha_l 1

# REDDIT-B 
# python main.py --lr 0.00015 --weight_decay 0.0 --max_epoch 100 --mask_rate 0.5 --drop_edge_rate 0.0 --num_hidden 512 --num_layers 2 --encoder gcn --decoder gcn --activation prelu --pooling sum --batch_size 8 --replace_rate 0.1 --norm layernorm --loss_fn sce --alpha_l 2

# COLLAB 
# python main.py --device 0 --lr 0.00015 --weight_decay 0.0 --max_epoch 20 --num_layers 2 --num_hidden 256 --mask_rate 0.75 --drop_edge_rate 0.0 --activation relu --encoder gcn --decoder gcn --pooling max --batch_size 32 --loss_fn sce --norm batchnorm --alpha_l 1
# python main.py --device 0 --lr 0.00015 --weight_decay 0.0 --max_epoch 20 --num_layers 2 --num_hidden 256 --mask_rate 0.5 --drop_edge_rate 0.0 --activation relu --encoder gcn --decoder gcn --pooling max --batch_size 32 --loss_fn sce --norm batchnorm --alpha_l 1

# seed = 0, 81.58 Version
# COLLAB
# python main.py --lr 0.00015 --weight_decay 0.0 --max_epoch 20 --num_layers 2 --num_hidden 256 --mask_rate 0.5 --drop_edge_rate 0.0 --activation relu --encoder gcn --decoder gcn --pooling max --loss_fn sce --norm batchnorm --alpha_l 1 --batch_size 64 
