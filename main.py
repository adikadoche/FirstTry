# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile, pstats
import pandas as pd
import io

import logging
from cli import parse_args
import torch
from pytorch_lightning.loggers import WandbLogger

# from modeling import Adi
from datetime import datetime
from training import set_seed, train


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    if "JOB_NAME" in os.environ:
        run_name = os.environ["JOB_NAME"]
    else:
        run_name = 'vscode'
    if args.resume_from:
        args.output_dir = args.resume_from
    else:
        args.output_dir = os.path.join(args.output_dir, datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S")+'_'+run_name)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    if not args.is_debug:
        wandb = WandbLogger(project='coref-detr', entity='adizicher', name=run_name)

    # Setup CUDA, GPU & distributed training
    if args.is_debug:
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = 1 
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    if args.is_debug:
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = 1 
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    wb_config = {}
    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")
        wb_config[key] = val
    if "GIT_HASH" in os.environ:
        wb_config["GIT_HASH"] = os.environ["GIT_HASH"]
    if not args.is_debug:
        wandb.experiment.config.update(wb_config)
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    if not args.is_debug:
        train(args, wandb)
    else:
        train(args)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
