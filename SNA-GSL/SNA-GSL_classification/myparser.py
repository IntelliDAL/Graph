import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device",default='cuda:0')
parser.add_argument("--seed",default=2022)
parser.add_argument("--explain_study",default=False)
# ----- hyper parameters -------
parser.add_argument('--sim_mat_learning_ablation',default=False)
parser.add_argument('--msa_bias',default=True)
# GCN layers
parser.add_argument("--embedding_size",default=32)
parser.add_argument("--graph_transformer_active",default=True)
parser.add_argument("--encoder_ffn_size",default=128)
parser.add_argument("--GT_res",default=True)
parser.add_argument("--share_qk",default=False)
parser.add_argument("--use_dist",default=True)
parser.add_argument("--dist_start_decay", type=float,default=0.5)

# GraphTransformer params
parser.add_argument("--n_heads", type=int,default=8)
parser.add_argument("--GCA_n_heads", type=int,default=8)
parser.add_argument('--channel_align',default=True)
parser.add_argument("--n_channel_transformer_heads", type=int,default=8)
parser.add_argument("--channel_ffn_size",default=128)

# conv params
parser.add_argument("--conv_channels_0", default=32)
parser.add_argument("--conv_channels_1", default=64)
parser.add_argument("--conv_channels_2", default=1)
parser.add_argument("--conv_channels_3", default=256)

parser.add_argument("--conv_l_relu_slope", default=0.33)
parser.add_argument("--pooling_res", default=20)
parser.add_argument("--conv_dropout", default=0.2)

# training parameters
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument('--iterations', type=int, help='number of training epochs', default=6000)
parser.add_argument('--iter_val_start', type=int, default=5400)
parser.add_argument('--patience', default=100)
parser.add_argument('--iter_val_every', type=int, default=1)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, help="Learning rate.", default=5e-4)
parser.add_argument("--lr_reduce_factor", default=0.5)
parser.add_argument("--lr_schedule_patience", default=1000)
parser.add_argument("--min_lr", default=1e-7)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--temp", default={'cur_iter': 0})

# experiment settings
parser.add_argument('--log_path', type=str, help='path for log file', default='../GSTLogs')
parser.add_argument('--repeat_run', type=int, help='indicated the index of repeat run', default=0)
parser.add_argument('--data_dir', type=str, help='root directory for the data', default='../datasets/')

# 预训练
parser.add_argument("--pre_epochs", type=int, default=15)
parser.add_argument('--pre_lr', type=float, default=0.0001)
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--aug", type=str, default='mask_nodes',
                    help='dnodes,pedges,subgraph,mask_nodes,random2,random3,random4')
parser.add_argument("--y", default=0.4)
parser.add_argument("--fc_embeddings", type=int, default=128)
parser.add_argument("--load_pre_model", default=0)
parser.add_argument("--load_GTCmodel", default=0)
parser.add_argument("--pre_train", default=0)
parser.add_argument('--dataset', type=str, default='AIDS',
                    help="IMDB-MULTI,IMDB-BINARY,AIDS,NCI1,ER_MD,COX2_MD,PROTEINS,REDDIT-BINARY")
parser.add_argument("--wandb_activate",default=1)
parsed_args = parser.parse_args()
if parsed_args.dataset == 'IMDB-BINARY':
    parsed_args.lr = 0.0001
    parsed_args.iterations = 4000
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4
    parsed_args.embedding_size = 32

if parsed_args.dataset == 'IMDB-MULTI':
    parsed_args.lr = 0.0001
    parsed_args.iterations = 4000
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4
if parsed_args.dataset == 'PROTEINS':
    parsed_args.lr = 0.0001
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4
if parsed_args.dataset == 'AIDS':
    parsed_args.lr = 0.0001
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4
if parsed_args.dataset == 'NCI1':
    parsed_args.lr = 0.0001
    parsed_args.pre_lr = 0.0001
    parsed_args.iterations = 5000
    parsed_args.n_heads = 2
    parsed_args.GCA_n_heads = 2
    parsed_args.n_channel_transformer_heads = 4
