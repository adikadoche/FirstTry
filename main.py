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
import wandb 
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

# from modeling import Adi
from datetime import datetime
from detr import build_DETR
from training import set_seed, train
from eval import make_evaluation, report_eval
from data import get_dataset, get_data_objects
from utils import load_from_checkpoint
from coref_bucket_batch_sampler import BucketBatchSampler


logger = logging.getLogger(__name__)


def create_optimization(model, args, train_loader):
    no_decay = ['bias', 'LayerNorm.weight']
    back_params = [args.backbone_name]

    model_decay = [p for n, p in model.named_parameters() if
                   any(hp in n for hp in back_params) and not any(nd in n for nd in no_decay)]
    model_no_decay = [p for n, p in model.named_parameters() if
                      any(hp in n for hp in back_params) and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in model.named_parameters() if
                  not any(hp in n for hp in back_params) and not any(nd in n for nd in no_decay)]
    head_no_decay = [p for n, p in model.named_parameters() if
                     not any(hp in n for hp in back_params) and any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {'params': model_decay, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': model_no_decay, 'lr': args.lr_backbone, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': head_no_decay, 'lr': args.lr, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr) #TODO check betas and epsilon
                    #   betas=(args.adam_beta1, args.adam_beta2),
                    #   eps=args.adam_epsilon)

    if args.max_steps > 0:
        args.t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        args.t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps / args.train_batch_size))
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps // args.train_batch_size,
                                                num_training_steps=args.t_total)
    # lr_scheduler = WarmupExponentialSchedule(optimizer, warmup_steps=int(args.warmup_steps / args.train_batch_size),
    #                                     gamma=0.99998)  # ConstantLRSchedule(optimizer)

    return optimizer, lr_scheduler


def main():
    args = parse_args()
    if "JOB_NAME" in os.environ:
        args.run_name = os.environ["JOB_NAME"]
    else:
        args.run_name = 'vscode'
    if args.resume_from:
        args.output_dir = args.resume_from
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S")+'_'+args.run_name)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    if not args.is_debug:
        wandb.init(project='coref-detr', entity='adizicher', name=args.run_name)

    # Setup CUDA, GPU & distributed training
    if args.is_debug:
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = 1
            os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
            os.environ["CUDA_VISIBLE_DEVICES"] = "7"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if not args.is_debug:    
        wb_config = wandb.config
        for key, val in vars(args).items():
            logger.info(f"{key} - {val}")
            wb_config[key] = val
        if "GIT_HASH" in os.environ:
            wb_config["GIT_HASH"] = os.environ["GIT_HASH"]
    else:
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
    if not args.is_debug:    
        wandb.watch(model, criterion, log="all")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir, add_prefix_space=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, add_prefix_space=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    
    eval_dataset = get_dataset(args, tokenizer, evaluate=True)
    eval_loader = BucketBatchSampler(eval_dataset, max_total_seq_len=args.max_total_seq_len, batch_size_1=True, n_gpu=1)

    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=False)
        train_loader = BucketBatchSampler(train_dataset, max_total_seq_len=args.max_total_seq_len, batch_size_1=args.batch_size_1, n_gpu=args.n_gpu)

        optimizer, lr_scheduler = create_optimization(model, args, train_loader)

        thresh_delta = 0.2    
        coref_threshold, cluster_threshold = 0.5, 0.5 # starting threshold, later fixed by eval
        best_f1 = -1
        best_f1_global_step = -1
        last_saved_global_step = -1
        if args.resume_from:
            logger.info("Loading from checkpoint {}".format(args.resume_from))
            loaded_args = load_from_checkpoint(model, args.resume_from, args.device, optimizer, lr_scheduler)
            coref_threshold = loaded_args['numbers']['coref_threshold']
            cluster_threshold = loaded_args['numbers']['cluster_threshold']
            best_f1_global_step = loaded_args['numbers']['best_f1_global_step']
            last_saved_global_step = loaded_args['numbers']['last_saved_global_step']
            best_f1 = loaded_args['numbers']['best_f1']
            thresh_delta = loaded_args['numbers']['thresh_delta']   

            if args.reset_optim:
                optimizer, lr_scheduler = create_optimization(model, args, train_loader)
                args.resume_global_step = 0
            else:
                args.resume_global_step = int(loaded_args['epoch']) * train_loader.total_len()
                args.num_train_epochs -= int(loaded_args['epoch'])
            if not args.do_train:
                return args.resume_global_step
        # if args.do_profile:
        #     profiler = cProfile.Profile()
        #     profiler.enable()
        #     global_step = train(args, model, criterion, train_loader, eval_loader, eval_dataset)
        #     profiler.disable()
        #     result = io.StringIO()
        #     pstats.Stats(profiler,stream=result).sort_stats('tottime').print_stats()
        #     result=result.getvalue()
        #     # chop the string into a csv-like buffer
        #     result='ncalls'+result.split('ncalls')[-1]
        #     result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
        #     # save it to disk
            
        #     with open('test.csv', 'w+') as f:
        #         #f=open(result.rsplit('.')[0]+'.csv','w')
        #         f.write(result)
        #         f.close()
        # else:
        global_step = train(args, model, criterion, train_loader, eval_loader, eval_dataset, optimizer, lr_scheduler, thresh_delta, coref_threshold, cluster_threshold,\
            best_f1_global_step, best_f1, last_saved_global_step)
        make_evaluation(model, criterion, eval_loader, eval_dataset, args) #TODO: report_eval won't work in here because of missing parameters
    else:
        loaded_args = load_from_checkpoint(model, args.resume_from, args.device)
        coref_threshold = loaded_args['numbers']['coref_threshold']
        cluster_threshold = loaded_args['numbers']['cluster_threshold']
        thresh_delta = loaded_args['numbers']['thresh_delta']   
        global_step = int(loaded_args['global_step'])
        results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, 0.5, 0.5, 0.2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
