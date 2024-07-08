CUDA_INDEX = 0
NAME = 'GED_AIDS700nef'
CLASSES = 29
import sys

sys.path.insert(0, '../..')
sys.path.insert(0, '../../pyged/lib')
import os

import time

import torch

torch.cuda.set_device(CUDA_INDEX)
torch.backends.cudnn.benchmark = True
import torch.optim
import torch_geometric as tg

from neuro import config, datasets, metrics, models, train, utils, viz
import pyged

from importlib import reload
import torch
print(torch.__version__)
print(torch.version.cuda)
reload(config)
reload(datasets)
reload(metrics)
reload(models)
reload(pyged)
reload(train)
reload(utils)
reload(viz)

train_set, train_meta = torch.load(f'../greed-data-and-models/data/{NAME}/train.pt', map_location='cpu')
# train_set, train_meta = torch.load(f'../greed-data-and-models/data/{NAME}/AIDS700nef_training.pt', map_location='cpu')
# viz.plot_inner_dataset(train_set, n_items=5, random=True)

val_set, _ = torch.load(f'../greed-data-and-models/data/{NAME}/val.pt', map_location='cpu')
# val_set, _ = torch.load(f'../greed-data-and-models/data/{NAME}/AIDS700nef_test.pt', map_location='cpu')
model = models.NormGEDModel(8, CLASSES, 64, 64).to(config.device)
loader = tg.loader.DataLoader(list(zip(*train_set)), batch_size=200, shuffle=True)
val_loader = tg.loader.DataLoader(list(zip(*val_set)), batch_size=1000, shuffle=True)

dump_path = os.path.join(f'../greed-data-and-models/runlogs/{NAME}', str(time.time()))
os.mkdir(dump_path)
train.train_full(model, loader, val_loader, lr=1e-3, weight_decay=1e-3,
                 cycle_patience=5, step_size_up=2000, step_size_down=2000, dump_path=dump_path)

