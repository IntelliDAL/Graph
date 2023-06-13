import datetime
import json

import wandb
from torch_geometric.data import Data as gdata, Batch
from myparser import parsed_args
import sys
import torch
import numpy as np
from timeit import default_timer as timer
from TU import GraphSimDataset
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

sys.path.append("")
print(sys.path)
import random
from aug import get
from utils import write_log_file, create_dir_if_not_exists
from BrainUSL import unsupervisedGroupContrast, Model, sameLoss
from model.GTC import GTCNet
from model.model import Transformer_Coss_Encoder
import torch.nn.functional as F
import os
from utils import calculate_metrics


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


args = parsed_args


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        setup_seed(self.args.seed)
        pre_train = 'pre-train_'
        self.f = 'logs/' + pre_train + args.dataset + '_' + args.aug + '.txt'
        # self.args.premodel_save_path = os.path.join("point/",
        #                                             pre_train + self.args.dataset + '_pre_' + 'GSA_heads_' + str(
        #                                                 self.args.n_heads) + '_GCA_heads_' + str(
        #                                                 self.args.GCA_n_heads) + 'model.pt')
        self.args.premodel_save_path = os.path.join("point/", '137_136_' + self.args.dataset + 'model.pt')
        self.args.model_save_path = os.path.join("point/",
                                                 pre_train + self.args.dataset + '_Decode_' + 'channel_heads' + str(
                                                     self.args.n_channel_transformer_heads) + '_model.pt')
        self.dataset = GraphSimDataset(self.args)
        self.BrainUSLModel = Model()
        # self.training_graphs = TUDataset(self.path, name=self.args.dataset, aug=args.aug).shuffle()
        # self.testing_graphs = TUDataset(self.path, name=self.args.dataset, aug='none').shuffle()
        self.args.n_max_nodes = self.dataset.n_max_nodes
        self.args.num_classes = self.dataset.training_graphs.num_classes
        try:
            self.args.in_features = self.dataset.training_graphs.num_features
        except:
            self.args.in_features = 1
        args_pre = args
        # args_pre.in_features = 137
        # args_pre.n_max_nodes = 136
        self.model = Transformer_Coss_Encoder(args_pre).to(args.device)

        write_log_file(self.f,
                       '------------------------------------------\n'
                       'topkPooling\n'
                       '{}\n'
                       'dataset {}, pre_epochs {}, lr {}, num_features {}, max nodes {} \n'.format(
                           self.f,
                           args.dataset, self.args.pre_epochs, self.args.lr, self.args.in_features,
                           self.args.n_max_nodes))

    def pre_train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.pre_lr)
        if self.args.load_pre_model:
            print('load model ...')
            self.model.load_state_dict(torch.load(self.args.premodel_save_path))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.args.lr_reduce_factor,
                                                               patience=self.args.lr_schedule_patience,
                                                               min_lr=self.args.min_lr,
                                                               verbose=True)
        for epoch in range(1, self.args.pre_epochs + 1):
            tic = timer()
            loss_all = 0
            self.model.train()
            batches = self.dataset.create_batches(self.dataset.training_set)

            for index, data in enumerate(batches):
                optimizer.zero_grad()

                G1, G1_aug = get(self.args.aug, data[0])
                G2, G2_aug = get(self.args.aug, data[1])
                G1 = self.dataset.transform_single(G1)
                G2 = self.dataset.transform_single(G2)
                G1_aug = self.dataset.transform_single(G1_aug)
                G2_aug = self.dataset.transform_single(G2_aug)

                x1 = G1['x']
                x1 = torch.matmul(x1, x1.transpose(1, 2))
                x2 = G2['x']
                x2 = torch.matmul(x2, x2.transpose(1, 2))
                sim_matrix1 = self.BrainUSLModel.get_label_matrix_from_sim(G1['adj'], y=self.args.y)
                sim_matrix2 = self.BrainUSLModel.get_label_matrix_from_sim(x1, y=self.args.y)
                sim_matrix_1 = sim_matrix1 & sim_matrix2
                sim_matrix1 = self.BrainUSLModel.get_label_matrix_from_sim(G2['adj'], y=self.args.y)
                sim_matrix2 = self.BrainUSLModel.get_label_matrix_from_sim(x2, y=self.args.y)
                sim_matrix_2 = sim_matrix1 & sim_matrix2
                # print('比例：', sim_matrix_1.sum() / (sim_matrix_1.shape[0] ** 2))
                # print('比例：', sim_matrix_2.sum() / (sim_matrix_2.shape[0] ** 2))
                y_pred_1, y_pred_2 = self.model(G1, G2)
                y_pred_1_aug, y_pred_2_aug = self.model(G1_aug, G2_aug)

                # loss_cal1 = self.model.loss_cal(y_pred_1, y_pred_1_aug)
                # loss_cal2 = self.model.loss_cal(y_pred_2, y_pred_2_aug)
                same_Loss1 = sameLoss(y_pred_1, y_pred_1_aug)
                same_Loss2 = sameLoss(y_pred_2, y_pred_2_aug)
                unsupervisedGroupContrast_loss1 = unsupervisedGroupContrast(y_pred_1, y_pred_1_aug, sim_matrix_1,
                                                                            self.args.T)
                unsupervisedGroupContrast_loss2 = unsupervisedGroupContrast(y_pred_2, y_pred_2_aug, sim_matrix_2,
                                                                            self.args.T)
                loss = 1.0 * (same_Loss1 + same_Loss2) / 2 + 1 * (
                        unsupervisedGroupContrast_loss1 + unsupervisedGroupContrast_loss2) / 2
                # loss=loss_cal1+loss_cal2
                loss_all += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
            # print(y_pred[0])
            toc = timer()
            write_log_file(self.f, 'Epoch {}, Loss {:.4f}, time:{:.2f}s'.format(epoch, loss_all / index, toc - tic))
            # if epoch % self.args.log_interval == 0:
            #     self.test()
        # print('比例：', sim_matrix_1.sum() / (sim_matrix_1.shape[0] ** 2))
        # print('比例：', sim_matrix_2.sum() / (sim_matrix_2.shape[0] ** 2))
        write_log_file(self.f,
                       ' dataset={},epoch={},val_epoch={},lr={},fc_embeddings={}\n'.format(
                           self.args.dataset,
                           self.args.pre_epochs,
                           self.args.log_interval,
                           self.args.lr,
                           self.args.fc_embeddings))
        torch.save(self.model.state_dict(), self.args.premodel_save_path)

    def train(self):
        self.GTCnet = GTCNet(self.args).to(self.args.device)
        print(self.GTCnet)
        # self.model.load_state_dict(torch.load(self.args.premodel_save_path))
        # self.model.load_state_dict(torch.load("point/137_136_IMDB-BINARYmodel.pt"))
        if self.args.load_GTCmodel:
            print('load GTC ...')
            self.GTCnet.load_state_dict(torch.load(self.args.model_save_path))
        GTCnet_optimizer = torch.optim.AdamW(self.GTCnet.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr * 0.001,
                                      weight_decay=self.args.weight_decay)
        GTCnet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(GTCnet_optimizer, mode='min',
                                                                      factor=self.args.lr_reduce_factor,
                                                                      patience=self.args.lr_schedule_patience,
                                                                      min_lr=self.args.min_lr,
                                                                      verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.args.lr_reduce_factor,
                                                               patience=self.args.lr_schedule_patience,
                                                               min_lr=self.args.min_lr,
                                                               verbose=True)
        min_loss = 1e10
        min_test_mse = 1e10
        max_acc_list = []
        max_auc_list = []
        max_f1_list = []
        for iteration in range(self.args.iterations):
            if self.args.pre_train:
                self.model.train()
            else:
                self.model.eval()
            tic = timer()
            batches = self.dataset.create_batches_(self.dataset.training_graphs)
            main_index = 0
            loss_sum = 0
            max_acc = 0
            max_auc = 0
            max_f1 = 0
            for index, batch_pair in enumerate(batches):
                GTCnet_optimizer.zero_grad()
                # optimizer.zero_grad()
                data = self.dataset.transform_single(batch_pair)
                #
                if self.args.pre_train:
                    x1, x2 = self.model.get_embeddings(data, data)
                else:
                    with torch.no_grad():
                        x1, x2 = self.model.get_embeddings(data, data)
                trues = data['target']
                prediction = self.GTCnet(x1, x2)

                loss = F.cross_entropy(prediction, trues, reduction='mean')
                loss.backward()
                GTCnet_optimizer.step()
                GTCnet_scheduler.step(loss)
                if self.args.pre_train:
                    optimizer.step()
                    scheduler.step(loss)
                loss_sum = loss_sum + loss.item()
                main_index = main_index + batch_pair.num_graphs

                pred = torch.argmax(prediction.cpu().detach(), dim=1)
                trues = trues.cpu().detach()

                acc = accuracy_score(trues, pred)
                if self.args.dataset == 'IMDB-MULTI':
                    aucs = roc_auc_score(trues, prediction.cpu().detach(), multi_class='ovo', average='macro')
                else:
                    aucs = roc_auc_score(trues, pred)
                f1, precision, recall = calculate_metrics(prediction.cpu().detach(), trues)
                # print(rercall)

                max_acc_list.append(acc)
                max_auc_list.append(aucs)
                max_f1_list.append(f1)

            loss = loss_sum / main_index
            acc = np.mean(max_acc_list)
            aucs = np.mean(max_auc_list)
            f1 = np.mean(max_f1_list)
            if iteration % max(1, self.args.iterations / 20) == 0:
                if (acc > max_acc):
                    max_acc = acc
                if (aucs > max_auc):
                    max_auc = aucs
                if f1 > max_f1:
                    max_f1 = f1
                toc = timer()
                write_log_file(self.f,
                               "Iteration = {}\tbatch loss = {} (e-3)\tacc = {} \tauc = {}\tf1={}\tmax_acc {} max_auc {} max f1 {} @ {}s" \
                               .format(iteration, loss * 1000, acc, aucs, f1, max_acc, max_auc, max_f1,
                                       toc - tic))
                if args.wandb_activate:
                    wandb.log(
                        {'epochs': iteration, 'CEloss': loss, 'train_acc': acc, 'train_auc': aucs, 'train_f1': f1})
                if loss < min_loss and iteration > 20:
                    min_loss = loss
                    # print('save  model...')
                    # torch.save(self.GTCnet.state_dict(), self.args.model_save_path)
            if (iteration + 1) % max(1, self.args.iterations // 10) == 0:
                test_mse = self.test()
                if min_test_mse > test_mse and iteration > 20:
                    min_test_mse = test_mse
                    print('save  model...')
                    torch.save(self.GTCnet.state_dict(), self.args.model_save_path)

    def test(self):
        tic = timer()
        print('\nModel evaluation.')
        self.GTCnet.eval()
        self.model.eval()
        batches = self.dataset.create_batches_(self.dataset.testing_graphs)
        loss_list = []
        acc_list = []
        auc_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        with torch.no_grad():
            for index, batch_pair in enumerate(batches):
                data = self.dataset.transform_single(batch_pair)
                x1, x2 = self.model.get_embeddings(data, data)
                trues = data['target']
                prediction = self.GTCnet(x1, x2)
                loss = F.cross_entropy(prediction, trues, reduction='mean')
                pred = torch.argmax(prediction.cpu().detach(), dim=1).numpy()
                trues = trues.cpu().detach()
                acc = accuracy_score(trues, pred)
                if self.args.dataset == 'IMDB-MULTI':
                    aucs = roc_auc_score(trues, prediction.cpu().detach(), multi_class='ovo', average='macro')
                else:
                    aucs = roc_auc_score(trues, pred)
                f1, precision, recall = calculate_metrics(prediction.cpu().detach(), trues)

                acc_list.append(acc)
                auc_list.append(aucs)
                f1_list.append(f1)
                precision_list.append(precision)
                recall_list.append(recall)
                loss_list.append(loss.item())

            loss = np.mean(loss_list)
            toc = timer()
            test_results = {
                'test_CE': loss,
                'test_acc': np.mean(acc_list),
                'test_auc': np.mean(auc_list),
                'test_f1': np.mean(f1_list),
                'test_precision': np.mean(precision_list),
                'test_recall': np.mean(recall_list),
                '@time': toc - tic
            }
            if args.wandb_activate:
                wandb.log(test_results)
            # write_log_file(self.f,"Test: CEloss = {}\nacc = {} \tauc = {}\tf1={} \tprecision = {}\trecall={} @ {}s\n" \
            #                .format(loss, np.mean(acc_list), np.mean(auc_list), np.mean(f1_list),np.mean(precision_list),np.mean(recall_list),
            #                        toc - tic))
            for k, v in test_results.items():
                write_log_file(self.f, '\t {} = {}'.format(k, v))
            return loss
