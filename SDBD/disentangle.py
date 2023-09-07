import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math, copy, time
import random
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold

from model2 import my_model
import logging
from utils import get_dis_args

runtime_id = 'disentangle-{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_path = './result/disentangle/{}.log'.format(runtime_id)
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

args = get_dis_args()

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
logger.info('device:{}'.format(args.gpu))

import warnings
warnings.filterwarnings("ignore")


X = np.load('./data/635_8A.npy')
logger.info('---------------------')
logger.info('X{}'.format(X.shape)) # N M M
logger.info('---------------------')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed = int(time.time()*2)
seed = 123
setup_seed(seed)
epochs = args.n_epoch
batch_size = 32
drop_out = 0.5
lr = 0.005
decay = 0.001
result = []

logger.info('a: {} b: {}'.format(args.a, args.b))

Model = my_model(args.a, args.b)
Model.to(device)

optimizer = torch.optim.Adam(Model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)

for epoch in range(0, epochs+1):
    Model.train()

    idx_batchs = np.random.permutation(int(X.shape[0]))
        
    for i in range(0,int(X.shape[0])//int(batch_size)):
        idx_batch = idx_batchs[i*int(batch_size):min((i+1)*int(batch_size), X.shape[0])]
        
        train_data_batch = X[idx_batch]
        train_data_batch = torch.from_numpy(train_data_batch).float().to(device)

        optimizer.zero_grad()
        A_loss , s_loss , d_loss = Model(train_data_batch)
        # A_loss = args.a * A_loss
        # s_loss = args.b * s_loss
        # d_loss = args.b * s_loss
        loss = A_loss + s_loss + d_loss
        loss.backward()
        optimizer.step()
            
    if epoch % 1 == 0:
        Model.eval()
            
        acc = 0
        cnt = 0
        for i in range(0,int(X.shape[0])//int(batch_size)):
            cnt += 1
            idx_batch = idx_batchs[i*int(batch_size):min((i+1)*int(batch_size), X.shape[0])]
            
            train_data_batch = X[idx_batch]
            train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                
            A_loss , s_loss , d_loss = Model(train_data_batch)
            # A_loss = args.a * A_loss
            # s_loss = args.b * s_loss
            # d_loss = args.b * s_loss  

        logger.info('train\tepoch: %d\tA_loss: %.4f\ts_loss: %.4f\td_loss: %.4f' % (epoch, A_loss, s_loss.item(), d_loss.item()))
        # torch.save(Model, './model_checkpoint/ABIDE_6_4/epochs{}-{}.pth'.format(epoch, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())))

torch.save(Model.state_dict(), './model_checkpoint/ABIDE2/epochs{}-a{:.2}-b{:.2}-{:.4}-{}.pth'.format(epochs, args.a, args.b, A_loss+s_loss+d_loss, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())))
# torch.save(Model, './model_checkpoint/epochs{}-{:.4}-{}.pth'.format(epochs, A_loss+s_loss+d_loss, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())))
logger.info('save in ./model_checkpoint/ABIDE2/epochs{}-a{:.2}-b{:.2}-{:.4}-{}.pth'.format(epochs, args.a, args.b, A_loss+s_loss+d_loss, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())))