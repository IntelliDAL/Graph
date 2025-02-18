import numpy as np
import torch
from telsrc.utils import def_args, set_random_seed, contr_cos_loss
from telsrc.data_util import load_regression_dataset
from telsrc.evaluation import graph_regression_evaluation
from telsrc.models.edcoder import define_tel

def pretrain(model, graph, optimizer, max_epoch, device, scheduler):
    graph = graph.to(device)
    model.train()
    for epoch in range(max_epoch):
        recPR_loss, (nembr, gembr), (nembw, gembw) = model(graph, graph.ndata["attr"])
        graph_contr_loss = contr_cos_loss(gembr, gembw, T=0.14)
        node_contr_loss = 0
        for i in range(0, len(nembr)):
            node_contr_loss = node_contr_loss + contr_cos_loss(nembr[i], nembw[i], T=0.14)
        loss = recPR_loss + (1-recPR_loss)*(node_contr_loss + graph_contr_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}|train_loss:{np.mean(loss.item()):.4f}")
    return model

def main(args):
    graph, (num_features, num_classes) = load_regression_dataset(args.dataset)
    args.num_features = num_features
    args.fea_len = num_features
    args.num_classes = num_classes
    test_list = []
    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        model = define_tel(args)
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        model = pretrain(model, graph, optimizer, args.max_epoch, args.device, scheduler)
        model.eval()
        test_result = graph_regression_evaluation(model, graph)
        test_list.append(test_result)
    final_test, final_test_std = np.mean(test_list), np.std(test_list)
    print(f"# final_test: {final_test:.4f}±{final_test_std:.4f}")

if __name__ == "__main__":
    args = def_args()
    main(args)

