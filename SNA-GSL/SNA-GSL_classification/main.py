import datetime

import torch
import wandb

from myparser import parsed_args
from train import Trainer

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parsed_args
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if args.wandb_activate:
        wandb.init(project='' + args.dataset, config=args, name=args.dataset + nowtime)
    trainer = Trainer(args)
    trainer.pre_train()
    trainer.train()
    wandb.finish()
