import os
import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl
from detr import DETRDataModule, DETR

logger = logging.getLogger(__name__)

def train(args, wandb=None):
    """ Train the model """
    model = DETR(args)
    data_model = DETRDataModule(args)

    if not args.is_debug:
        wandb.watch(model, log="all")    
    if args.local_rank == 0:
        torch.distributed.barrier()  #???????? # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)

    if wandb is not None:        
        # trainer = pl.Trainer(max_epochs=args.num_train_epochs, gpus=args.n_gpu, amp_backend='apex', logger= wandb, accumulate_grad_batches=args.gradient_accumulation_steps,\
        #     default_root_dir=args.output_dir, profiler=profiler)
        trainer = pl.Trainer(max_epochs=args.num_train_epochs, gpus=args.n_gpu, amp_backend='apex', logger= wandb, accumulate_grad_batches=args.gradient_accumulation_steps,\
            default_root_dir=args.output_dir)
    else:
        trainer = pl.Trainer(max_epochs=args.num_train_epochs, gpus=args.n_gpu, amp_backend='apex', accumulate_grad_batches=args.gradient_accumulation_steps,\
            default_root_dir=args.output_dir, detect_anomaly=True)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        print(args.output_dir)
        os.makedirs(args.output_dir)

    trainer.fit(model, data_model)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
