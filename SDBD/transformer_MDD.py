import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import random
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold

from model_MDD import my_model
from utils import get_args
import logging

args = get_args()

runtime_id = 'MDD-tansformer-{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_path = './result/{}/{}.log'.format(args.save, runtime_id)
fh = logging.FileHandler(file_path)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Create log file at {}'.format(file_path))


device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
logger.info('device:{}'.format(device))

import warnings
warnings.filterwarnings("ignore")

class E2E(nn.Module):

    def __init__(self, windows, in_channel, out_channel, input_shape, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.windows = windows
        
        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))

    def forward(self, A):
        
        A = A.view(-1, self.in_channel, 116, 116)
        a = self.conv1xd(A)
        b = self.convdx1(A)
        
        concat1 = torch.cat([a]*self.d, 2)
        concat2 = torch.cat([b]*self.d, 3)
        A = concat1+concat2
        
        return A


class BrainCNN(nn.Module):
    def __init__(self, f_length, windows, dropout=0.5, num_class=2):
        super().__init__()
        
        self.windows = windows
        self.f_length = f_length

        self.e2e = nn.Sequential(
            E2E(windows, 1, 8, (116, 116)),
            nn.LeakyReLU(0.33),
#             E2E(windows, 8, 8, (200, 200)),
#             nn.LeakyReLU(0.33),
#             E2E(8, 8, (200, 200)),
#             nn.LeakyReLU(0.33),
        )
        
        self.e2n2g = nn.Sequential(
            nn.Conv2d(8, 3, (1, 116)),
            nn.LeakyReLU(0.33),
            nn.Conv2d(3, 116, (116, 1)),
            nn.LeakyReLU(0.33),
        )

        self.linear = nn.Sequential(
            nn.Linear(116, f_length),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(0.33),
#             nn.Linear(64, 10),
#             nn.Dropout(dropout),
#             nn.LeakyReLU(0.33),
#             nn.Linear(10, num_class)
        )       

        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        x = self.e2e(x)
        x = self.e2n2g(x)
        x = x.view(-1, self.windows, 116)
        x = self.linear(x)
#         x = F.softmax(x, dim=-1)
#         print('output', x.shape)

        return x
        

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed, d_model1, d_model2):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.linear = nn.Sequential(
                            # nn.Linear(d_model1*d_model2*200, d_model1*d_model2),
                            # nn.Dropout(0.5),
                            # nn.ReLU(),
                            nn.Linear(d_model1*d_model2, 32),
                            nn.Dropout(0.5),
                            nn.ReLU(),
                            nn.Linear(32, 2)
                        )

        # self.BrainCNN = BrainCNN(d_model1, d_model2)
        # logger.info('use BrainCNN')

        # a = float(args.checkpoint[37:40])
        # b = float(args.checkpoint[42:45])
        a = 1.0
        b = 1.0
        logger.info('a: {} b: {}'.format(a, b))

        self.pre = my_model(a, b)
        self.pre.load_state_dict(torch.load(args.checkpoint))
        # self.pre = torch.load(args.checkpoint)
        logger.info('load model: {}'.format(args.checkpoint))

        self.fold = nn.Sequential(
                            nn.Linear(116*8, d_model1),
                            nn.Dropout(0.5),
                            nn.ReLU()
                        )
        
    def forward(self, src, A, src_mask=None):
        "Take in and process masked src and target sequences."
        # print(src.shape)
        # src = self.BrainCNN(src)
        
        src = self.pre.Encoder(src)
        src = src.view(src.size(0), src.size(1), -1)
        src = self.fold(src)

        src, loss = self.encode(src, A, src_mask)
        src = src.view(src.size(0), -1)
        src = self.linear(src)
        src = F.softmax(src, dim=-1)
        return src, loss
    
    def encode(self, src, A, src_mask):
        return self.encoder(self.src_embed(src), A, src_mask)
#         return self.encoder(src, src_mask)   

    def diff(self, src):
        difflist = list()
        for i in range(1,8):
            difflist.append((src[:,i,:] - src[:,i-1,:]))
        for i in range(1,8):
            src[:,i,:] = src[:,i,:] + difflist[i-1]
        return src
        

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, A, mask):
        "Pass the input (and mask) through each layer in turn."
        loss = 0
        for i, layer in enumerate(self.layers):
            x, t_loss = layer(x, A, mask)
            loss += t_loss
        return self.norm(x), loss

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if A is None:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            x0 = x
            x, loss = sublayer(self.norm(x), A)
            return x0 + self.dropout(x), loss

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, size2, self_attn1, self_attn2, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn1 = self_attn1
        self.self_attn2 = self_attn2
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer2 = clones(SublayerConnection(size2, dropout), 1)
        self.size = size

    def forward(self, x, A, mask):
        "Follow Figure 1 (left) for connections."
        x, loss = self.sublayer[0](x, A, lambda x, A: self.self_attn1(x, x, x, A, mask))
        return self.sublayer[1](x, None, self.feed_forward), loss

def attention(query, key, value, A, h, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    scores = torch.matmul(query, key.transpose(-2, -1))
    p_attn = F.softmax(scores, dim = -1)
#     A = A.view(-1, 1, 200, 200)
#     A = A.repeat(1, h, 1, 1)
#     print(A.shape, p_attn.shape, (p_attn-A).shape)
#     loss = (p_attn-A).abs().sum() / h * 0.0001
#     loss = Aloss_fn(p_attn, A) / h * 0.001
    loss = 0
#     print(loss)
    return torch.matmul(p_attn, value), loss

def attention2(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    scores = torch.matmul(key.transpose(-2, -1), value)
    return torch.matmul(query, scores)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # print(d_model, h)
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # (3 + 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.ELU = nn.ELU()
        
    def forward(self, query, key, value, A, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        # x = attention2(self.ELU(query)+1, self.ELU(key)+1, value, mask=mask, dropout=self.dropout)
        x, loss = attention(query, key, value, A, self.h, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), loss

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def make_model(N=6, d_model1=72, d_model2=200, d_ff=2048, h1=8, h2=10, dropout=0.5):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn1 = MultiHeadedAttention(h1, d_model1)
    attn2 = MultiHeadedAttention(h2, d_model2)
    ff = PositionwiseFeedForward(d_model1, d_ff, dropout)
    position = PositionalEncoding(d_model1, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model1, d_model2, c(attn1), c(attn2), c(ff), dropout), N),
                           c(position),
                           d_model1, d_model2)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


X = np.load('./data/MDD_HC_427_18A.npy')
Y = np.load('./data/MDD_HC_label.npy')

X = X[15:-15]
Y = Y[15:-15]

logger.info('---------------------')
logger.info('X{}'.format(X.shape)) # N M M
logger.info('Y{}'.format(Y.shape))
logger.info('---------------------')

# seed = int(time.time()*2)
seed = args.seed
setup_seed(seed)
epochs = args.n_epoch
batch_size = args.bs
drop_out = args.drop_out
lr = args.lr
decay = args.decay
result = []

logger.info('epochs = {}'.format(epochs))
logger.info('batch_size = {}'.format(batch_size))
logger.info('dropout = {}'.format(drop_out))
logger.info('lr = {}'.format(lr))
logger.info('decay = {}'.format(decay))
logger.info('seed = {}'.format(seed))

kf = KFold(n_splits=5, random_state=seed, shuffle=True)
kfold_index = 0
for train_index, test_index in kf.split(X):
    kfold_index += 1
    logger.info('kfold_index:{}'.format(kfold_index))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    logger.info('X_train{}'.format(X_train.shape))
    logger.info('X_test{}'.format(X_test.shape),)
    logger.info('Y_train{}'.format(Y_train.shape))
    logger.info('Y_test{}'.format(Y_test.shape))
#     print(Y_test)

    # model
    Model = make_model(2, 64, 18, 128, 2, 2)
    Model.to(device)
#     for p in Model.parameters():
#         if p.requires_grad:
#             print(p.name, p.data.shape)
    
#     optimizer = torch.optim.SGD(Model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()), 
#                                   lr=lr,
#                                   weight_decay=decay)
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    loss_fn = nn.CrossEntropyLoss()
    Aloss_fn = nn.MSELoss(reduction='sum')
    
    # train
    for epoch in range(0, epochs+1):
        Model.train()

        idx_batchs = np.random.permutation(int(X_train.shape[0]))
        
        for i in range(0,int(X_train.shape[0])//int(batch_size)):
            idx_batch = idx_batchs[i*int(batch_size):min((i+1)*int(batch_size), X_train.shape[0])]
        
            train_data_batch = X_train[idx_batch]
            train_label_batch = Y_train[idx_batch]
            train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
            train_label_batch = torch.from_numpy(train_label_batch).long()

            optimizer.zero_grad()
            outputs, losses = Model(train_data_batch, 0)
#             print(outputs.shape)
            outputs = outputs.cpu()
            loss = F.cross_entropy(outputs, train_label_batch, reduction='mean')
#             losses = losses.cpu()
#             loss += losses
            loss.backward()
            optimizer.step()
            
        if epoch % 5 == 0:
            Model.eval()
            
            acc = 0
            cnt = 0
            for i in range(0,int(X_train.shape[0])//int(batch_size)):
                cnt += 1
                idx_batch = idx_batchs[i*int(batch_size):min((i+1)*int(batch_size), X_train.shape[0])]
            
                train_data_batch = X_train[idx_batch]
                train_label_batch = Y_train[idx_batch]
                train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch = torch.from_numpy(train_label_batch).long()
                
                outputs, losses = Model(train_data_batch, 0)
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                # print(preds)
                acc += metrics.accuracy_score(preds, train_label_batch)
            logger.info('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f' % (epoch, loss.item(),acc/cnt))
            
        # test
        if epoch % 5 == 0:
            Model.eval()
            
            test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
            outputs, _ = Model(test_data_batch_dev, 0)
            _, indices = torch.max(outputs, dim=1)
            preds = indices.cpu()
            # print(preds)
            acc = metrics.accuracy_score(Y_test, preds)
            precision = metrics.precision_score(Y_test, preds)
            reacall = metrics.recall_score(Y_test, preds)
            f1 = metrics.f1_score(Y_test, preds)
            auc = metrics.roc_auc_score(Y_test, preds)
            logger.info('test\tacc: %.4f\tprecision: %.4f\trecall: %.4f\t\tf1: %.4f\t\tauc: %.4f' % (acc, precision, reacall, f1, auc))

            result.append([epoch, acc, precision, reacall, f1, auc])

result_epoch_list = [30, 50, 70, 100]
num = len(result) // 5
for i in range(num):
    if (result[i][0] not in result_epoch_list):
        continue

    logger.info(result[i])
    logger.info(result[i+num])
    logger.info(result[i+2*num])
    logger.info(result[i+3*num])
    logger.info(result[i+4*num])
    acc = (result[i][1]+result[i+num][1]+result[i+2*num][1]+result[i+3*num][1]+result[i+4*num][1]) / 5
    precision = (result[i][2]+result[i+num][2]+result[i+2*num][2]+result[i+3*num][2]+result[i+4*num][2]) / 5
    reacall = (result[i][3]+result[i+num][3]+result[i+2*num][3]+result[i+3*num][3]+result[i+4*num][3]) / 5
    f1 = (result[i][4]+result[i+num][4]+result[i+2*num][4]+result[i+3*num][4]+result[i+4*num][4]) / 5
    auc = (result[i][5]+result[i+num][5]+result[i+2*num][5]+result[i+3*num][5]+result[i+4*num][5]) / 5
    logger.info('epoch:{}'.format(result[i][0]))
    logger.info('acc:{:.6}'.format(acc))
    logger.info('precision:{:.6}'.format(precision))
    logger.info('recall:{:.6}'.format(reacall))
    logger.info('f1:{:.6}'.format(f1))
    logger.info('auc:{:.6}'.format(auc))
    logger.info('\n')


# with open("./result/result.txt","a") as file:
#     file.write('{}-{}-{}-{}-{:.4}\n'.format(epochs, lr, batch_size, decay, acc/5))