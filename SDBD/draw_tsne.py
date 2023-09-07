# from dual_gnn.dataset.DomainData import DomainData
# from graphmae.models import build_model
# from graphmae.utils import build_args, load_best_configs, set_random_seed
# import dgl
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt

from model_BP import my_model

# from rename_UDA import GNN


def data_split(y,train_size, val_size):
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        set_random_seed(seed)
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * train_size)
    val_size = int(len(random_node_indices) * val_size)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size + val_size:]
    train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    val_masks[val_node_indices] = 1
    test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    return train_masks,val_masks,test_masks

def plot_embedding(result, label, path, index):
    # x_min, x_max = torch.min(result, 0), np.max(result, 0)  # 分别求出每一列最小值和最大值
    # data = (result - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
    # result = data
    plt.figure(figsize=(10, 8))  # 创建一个画布
    # central_kind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    central_kind = [0, 1]
    # central_kind = ['UCLA', 'UM', 'USM','NYU']
    data = {}
    for central_kind_ in range(len(central_kind)):
        data_ = []
        for i in range(label.shape[0]):
            if label[i] == central_kind_:
                # if -4 < result[i][0] < 5 and -4 < result[i][1] < 5:  # original graph
                    # if result[i][0] < 10 and result[i][1] < 25: #gen graph (2,3)
                    # if -200 <result[i][0] < 200 and -200 <result[i][1] < 250: #gen graph (3,4)
                    # data_.append(result[i])
                data_.append(result[i])

        data[central_kind_] = data_
    # colors = ['black', 'red', 'green', '#FF00FF', 'blue']
    colors = ['black', 'red', 'green', '#FF00FF', 'blue','black', 'red', 'green', '#FF00FF', 'blue']
    markers = [',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*', 'h', '+', '*', 'x']
    for k, v in data.items():
        '''target features'''
        # plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=20, marker='o', c=colors[k], label=central_kind[k])
        '''raw features'''
        # if 0<= k <= 4:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='o', c=colors[k], label=central_kind[k])
        # else:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='*', c=colors[k], label=central_kind[k],alpha = 0.6)
        '''UDA features'''
        if k > 4:
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=30, marker='o', c=colors[k], label=central_kind[k])
        else:
            # plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=10, marker='o', c=colors[k], label=central_kind[k],alpha = 0.3)
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=30, marker='*', c=colors[k], label=central_kind[k], alpha = 0.6)

        # if k == 0:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] + 10, s=10, marker='o', c=colors[k], label=central_kind[k])
        # elif k == 5:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1]+ 10, s=10, marker='*', c=colors[k], label=central_kind[k],
        #                 alpha=0.6)
        # elif k == 1:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] + 10, s=10, marker='o', c=colors[k], label=central_kind[k])
        # elif k == 6:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] + 10, s=10, marker='*', c=colors[k], label=central_kind[k],
        #                 alpha=0.6)
        # elif k ==2:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] -20, s=10, marker='o', c=colors[k], label=central_kind[k])
        # elif k == 7:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] - 20, s=10, marker='*', c=colors[k], label=central_kind[k],
        #                 alpha=0.6)
        # elif k == 3:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] - 20, s=10, marker='o', c=colors[k], label=central_kind[k])
        # elif k == 8:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1]- 20, s=10, marker='*', c=colors[k], label=central_kind[k],
        #                 alpha=0.6)
        # elif k == 4:
        #     plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1] - 20, s=10, marker='o', c=colors[k], label=central_kind[k])
        # elif k == 9:
        #     plt.scatter(np.array(v)[:, 0] , np.array(v)[:, 1]- 20, s=10, marker='*', c=colors[k], label=central_kind[k],
        #                 alpha=0.6)
        # plt.legend(loc=2, bbox_to_anchor=(1.0,1.0),prop = {'size':8})

    plt.xticks(())  # 不显示坐标刻度
    plt.yticks(())
    # plt.title(title)  # 设置标题
    # plt.savefig(str(1) + '_' + 't-SNE.png')
    '''raw_features'''
    # plt.savefig('results/figures/raw_features.png')
    '''UDA_features'''
    # plt.savefig('results/figures/UDA_target.png')
    # plt.savefig('results/figures/UDA_features.png')
    '''SSL_features'''
    # plt.savefig('results/figures/SSL_features.png')
    '''Source_only_features'''
    # plt.savefig('results/figures/Source_only_features.png')
    '''UDAGCN'''
    # plt.savefig('results/figures/UDAGCN.png')
    '''node alignmet and classfication'''
    # plt.savefig('results/figures/alignment_class.png')
    plt.savefig('result/figures/tsne{:d}.png'.format(index))
    plt.show()


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    target = args.target
    source = args.source
    args.num_features = 6775

    # graph, (num_features, num_classes) = load_dataset(dataset_name)
    dataset_target = DomainData("data_/{}".format(args.target), name=target)
    dataset_source = DomainData("data_/{}".format(args.source), name=source)
    # print('dataset_target.edge_index',dataset_target[0].edge_index, 'type(index): ',type(dataset_target[0].edge_index))
    #
    target_data = dataset_target[0]
    t_src = target_data.edge_index[0]
    t_dst = target_data.edge_index[1]
    graph_target = dgl.graph((t_src, t_dst))
    graph_target = dgl.to_bidirected(graph_target)
    graph_target = graph_target.remove_self_loop().add_self_loop()
    graph_target.create_formats_()

    source_data = dataset_source[0]
    s_src = source_data.edge_index[0]
    s_dst = source_data.edge_index[1]
    graph_source = dgl.graph((s_src, s_dst))
    graph_source = dgl.to_bidirected(graph_source)
    graph_source = graph_source.remove_self_loop().add_self_loop()
    graph_source.create_formats_()

    # graph = preprocess(graph)
    '''target data split'''
    t_train_masks, t_val_masks, t_test_masks = data_split(y=target_data.y, train_size=0.7, val_size=0.1)
    # train_masks = torch.full((target_data.y.shape[0],), False).index_fill_(0, torch.as_tensor(train_node_indices[0]), True)
    print('graph_target: ', graph_target, ' graph_source: ', graph_source)
    print('target_data: ', target_data, ' source_data: ', source_data)
    # #
    graph_target.ndata['feat'] = target_data.x
    graph_target.ndata['label'] = target_data.y
    graph_target.ndata['train_mask'] = t_train_masks
    graph_target.ndata['val_mask'] = t_val_masks
    graph_target.ndata['test_mask'] = t_test_masks
    print('graph_target.ndatatrain_mask ', graph_target.ndata['train_mask'][:10], ' target_data.val_mask: ',
          graph_target.ndata['val_mask'][:10], ' target_data.test_mask: ', graph_target.ndata['test_mask'][:10])

    graph_source.ndata['feat'] = source_data.x
    graph_source.ndata['label'] = source_data.y

    # # '''raw features'''
    # raw_labels_t = graph_target.ndata['label'].add(5)
    # print('type(features): ',type(graph_target.ndata['feat']), ' shape: ',graph_target.ndata['feat'].shape,' raw_labels_t.shape: ',raw_labels_t.shape, ' type(raw_labels_t): ',type(raw_labels_t))
    # raw_features = torch.cat([graph_target.ndata['feat'],graph_source.ndata['feat']], 0)
    # raw_labels = torch.cat([raw_labels_t,graph_source.ndata['label'],], 0)
    #
    # # '''UDA'''
    model = build_model(args).to(device)
    # # '''UDA'''
    model.load_state_dict(torch.load('test_models/'+args.source+'_'+args.target+'_UDA_visual.pt'))
    # '''SSL'''
    # model.load_state_dict(torch.load('test_models/'+args.source+'_'+args.target+'_visual.pt'))
    '''source_only'''
    # model.load_state_dict(torch.load('test_models/' + args.source + '_' + args.target + '_Source_only_visual.pt'))
    encoder = model.encoder
    encoder.to(device)
    graph_target = graph_target.to(device)
    graph_source = graph_source.to(device)
    feat_target = graph_target.ndata['feat'].to(device)
    feat_source = graph_source.ndata['feat'].to(device)
    feat_target = encoder(graph_target, feat_target)
    feat_source = encoder(graph_source,feat_source )
    raw_labels_t = graph_target.ndata['label'].add(5).to(device)
    # print('type(features): ',type(graph_target.ndata['feat']), ' shape: ',graph_target.ndata['feat'].shape,' raw_labels_t.shape: ',raw_labels_t.shape, ' type(raw_labels_t): ',type(raw_labels_t))
    raw_features = torch.cat([feat_target,feat_source], 0)
    raw_labels = torch.cat([raw_labels_t,graph_source.ndata['label']], 0)
    # raw_features = torch.cat([feat_target, feat_target], 0)
    # raw_labels = torch.cat([raw_labels_t, raw_labels_t], 0)


    # '''UDAGCN'''
    # encoder = GNN(type="gcn").to(device)
    # encoder.load_state_dict(torch.load('test_models/' + args.source + '_' + args.target + '_UDAGCN_visual.pt'))
    # # encoder = model.encoder
    # # encoder.to(device)
    # graph_target = graph_target.to(device)
    # graph_source = graph_source.to(device)
    # feat_target = graph_target.ndata['feat'].to(device)
    # feat_source = graph_source.ndata['feat'].to(device)
    # feat_target = encoder(graph_target, feat_target)
    # feat_source = encoder(graph_source, feat_source)
    # raw_labels_t = graph_target.ndata['label'].add(5).to(device)
    # # print('type(features): ',type(graph_target.ndata['feat']), ' shape: ',graph_target.ndata['feat'].shape,' raw_labels_t.shape: ',raw_labels_t.shape, ' type(raw_labels_t): ',type(raw_labels_t))
    # raw_features = torch.cat([feat_target, feat_source], 0)
    # raw_labels = torch.cat([raw_labels_t, graph_source.ndata['label']], 0)

    return raw_features, raw_labels

if __name__ == '__main__':
    # args = build_args()
    # if args.use_cfg:
    #     args = load_best_configs(args, "configs.yml")
    # print(args)
    # data, label = main(args)

    # data, label, n_samples, n_features = get_data()  # data种保存[1083,64]的向量
    # data, label = get_embed_label_data()  # data种保存[1083,64]的向量

    data = np.load('./data/BP_HC_392_18A.npy')
    label = np.load('./data/BP_HC_label.npy')
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()

    pre = my_model()
    pre.load_state_dict(torch.load('./model_checkpoint/BP-epochs30-24.7-2023-03-13-21:42:50.pth'))

    data = pre.Encoder(data)
    data = data.view(data.size(0), data.size(1), -1, 16)
    data = data.mean(dim=2)

    print('data.shape: ',data.shape,' type(label): ',type(label))
    print('label: ',label)
    # data = data.reshape(data.shape[0],-1)
    data = data.cpu().detach().numpy()

# 没有pca，加pca效果不行，tsne用参数
# perplexity 5～17
# early_exaggeration 8～29
# learning_rate 3～20
# min_grad_norm 1e-8
# metric canberra，correlation，cosine，seuclidean
    tsne = TSNE(n_components=2, init='pca',
                random_state=123)  # n_components将64维降到该维度，默认2；init设置embedding初始化方式，可选pca和random，pca要稳定些
    for i in range(18):
        tmp_data = data[:,i,:]
        result = tsne.fit_transform(tmp_data)  # 进行降维，[1083,64]-->[1083,2]
        plot_embedding(result, label, '', i) #显示数据