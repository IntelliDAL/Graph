import os
import random
import datetime
import torch
from src2.parser import parsed_args
from trainer import GEDTrainer
from utils import create_dir_if_not_exists, tab_printer, log_args
import wandb
if __name__ == '__main__':

    if parsed_args.wandb_activate:

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=parsed_args.dataset, config=parsed_args, name=parsed_args.dataset + nowtime,resume="allow")

    torch.manual_seed(parsed_args.seed)
    if torch.cuda:
        torch.cuda.manual_seed(parsed_args.seed)

    if torch.cuda.device_count() == 1:
        parsed_args.gpu_index = '0'

    d = torch.device(('cuda:' + parsed_args.gpu_index) if torch.cuda.is_available() else 'cpu')
    parsed_args.device = d
    os.environ['CUDA_VISIBLE_DEVICES'] = parsed_args.gpu_index

    create_dir_if_not_exists(parsed_args.log_path)
    log_root_dir = parsed_args.log_path
    # signature = parsed_args.dataset + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") \
    #             + '-' + str(random.randint(100000, 1000000))
    signature=parsed_args.dataset + '_n_heads_' +str(parsed_args.n_heads)
    current_run_dir = os.path.join(log_root_dir, signature)
    create_dir_if_not_exists(current_run_dir)
    parsed_args.current_run_dir=current_run_dir
    # parsed_args.model_save_path = os.path.join(current_run_dir, 'best_model.pt')
    parsed_args.log_file_path = os.path.join(current_run_dir, 'log.txt')
    parsed_args.dist_mat_path = os.path.join(parsed_args.data_dir, parsed_args.dataset,
                                             parsed_args.dataset + '_distance.npy')
    ged_main_dir = parsed_args.data_dir

    tab_printer(parsed_args)
    log_args(parsed_args.log_file_path, parsed_args)
    trainer = GEDTrainer(args=parsed_args)

    # trainer.pre_train()
    trainer.train()
    trainer.test()
    wandb.finish()

    # if parsed_args.load_model:
    #     parsed_args.model_save_path = os.path.join(log_root_dir, parsed_args.loaded_model_signature, "best_model.pt")
    # else:
    #     trainer.train()
    #
    # if parsed_args.explain_study:
    #     trainer.explain_study()
    # else:
    #     trainer.test()
