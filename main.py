# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
from cli import parse_args
import torch
#TODO import wandb 

# from modeling import Adi
from detr import build_DETR
from training import set_seed, train
from eval import make_evaluation
from data import get_dataset, get_data_objects


logger = logging.getLogger(__name__)


def main():
    print("FML")
    args = parse_args()
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()


    # config_class = LongformerConfig
    # base_model_prefix = "longformer"
    model, criterion = build_DETR(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    train_dataset, train_sampler, train_loader, args.train_batch_size = get_data_objects(args, 'train.english.512.jsonlines', True)
    eval_dataset, eval_sampler, eval_loader, args.eval_batch_size = get_data_objects(args, 'dev.english.512.jsonlines', False)

    global_step = train(args, model, criterion, train_loader, eval_loader)
    make_evaluation(model, criterion, eval_loader, args)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
