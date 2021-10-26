# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cProfile, pstats
import pandas as pd
import io

import logging
from cli import parse_args
import torch
import wandb 

# from modeling import Adi
from datetime import datetime
from detr import build_DETR
from training import set_seed, train
from eval import make_evaluation
from data import get_dataset, get_data_objects


logger = logging.getLogger(__name__)
wandb.init(project='coref-detr', entity='adizicher')


def main():
    print("FML")
    args = parse_args()
    if args.resume_from:
        args.output_dir = args.resume_from
    else:
        args.output_dir = os.path.join(args.output_dir, datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S"))
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    if args.run_name != '':
        wandb.run.name = args.run_name

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
    args.n_gpu = 1   #TODO:REMOVEEEEEEEe

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    wb_config = wandb.config
    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")
        wb_config[key] = val
    if "GIT_HASH" in os.environ:
        wb_config["GIT_HASH"] = os.environ["GIT_HASH"]
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()


    # config_class = LongformerConfig
    # base_model_prefix = "longformer"
    model, criterion = build_DETR(args)
    wandb.watch(model, criterion, log="all")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    eval_dataset, eval_sampler, eval_loader, args.eval_batch_size = get_data_objects(args, args.predict_file, False)

    if args.do_train:
        train_dataset, train_sampler, train_loader, args.train_batch_size = get_data_objects(args, args.train_file, True)
        global_step = train(args, model, criterion, train_loader, eval_loader, eval_dataset)
    make_evaluation(model, criterion, eval_loader, eval_dataset, args) #TODO: report_eval won't work in here because of missing parameters

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    result = io.StringIO()
    pstats.Stats(profiler,stream=result).sort_stats('tottime').print_stats()
    result=result.getvalue()
    # chop the string into a csv-like buffer
    result='ncalls'+result.split('ncalls')[-1]
    result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
    # save it to disk
    
    with open('test.csv', 'w+') as f:
        #f=open(result.rsplit('.')[0]+'.csv','w')
        f.write(result)
        f.close()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
