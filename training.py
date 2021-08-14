import json
import os
import logging
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from tqdm import tqdm, trange
import time, datetime
from misc import save_on_master, is_main_process
from utils import reduce_dict


from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    epoch_iterator, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(tensor.to(device) for tensor in batch)
        input_ids, attention_mask, gold_clusters = batch
        model.train()
        criterion.train()
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        # header = 'Epoch: [{}]'.format(epoch)
        # print_freq = 10

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #     samples = samples.to(device)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(input_ids, attention_mask=attention_mask)
        loss_dict = criterion(outputs, gold_clusters)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train(args, train_dataset, model, tokenizer, criterion, evaluator):
    """ Train the model """
    # output_dir = Path(args.output_dir)

    tb_path = os.path.join(args.tensorboard_dir, os.path.basename(args.output_dir))
    tb_writer = SummaryWriter(tb_path, flush_secs=30)
    logger.info('Tensorboard summary path: %s' % tb_path)

    data_loader_train = BucketBatchSampler(train_dataset, max_total_seq_len=args.max_total_seq_len, batch_size_1=args.batch_size_1)

    t_total = len(data_loader_train) // args.gradient_accumulation_steps * args.num_train_epochs

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone, #TODO: learn how to freeze backbone
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    loaded_saved_optimizer = False
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        loaded_saved_optimizer = True

    # if args.amp:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #
    # # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #
    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    start_epoch = 0
    if os.path.exists(args.model_name_or_path) and 'checkpoint' in args.model_name_or_path:
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            start_epoch = int(checkpoint_suffix)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from global step %d", start_epoch)
            if not loaded_saved_optimizer:
                logger.warning("Training is continued from checkpoint, but didn't load optimizer and scheduler")
        except ValueError:
            logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    # If nonfreeze_params is not empty, keep all params that are
    # not in nonfreeze_params fixed.
    if args.nonfreeze_params:
        names = []
        for name, param in model.named_parameters():
            freeze = True
            for nonfreeze_p in args.nonfreeze_params.split(','):
                if nonfreeze_p in name:
                    freeze = False

            if freeze:
                param.requires_grad = False
            else:
                names.append(name)

        print('nonfreezing layers: {}'.format(names))

    best_f1 = -1
    best_epoch = -1
    start_time = time.time()
    for epoch in range(start_epoch, args.num_train_epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
        epoch_iterator = tqdm(data_loader_train, desc="Iteration", disable=args.local_rank not in [-1, 0])
        train_stats = train_one_epoch(
            model, criterion, epoch_iterator, optimizer, args.device, epoch)
        scheduler.step()
        if args.output_dir:
            checkpoint_paths = [args.output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(args.output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        results = evaluator.evaluate(model, prefix=f'step_{epoch}', tb_writer=tb_writer, global_step=epoch)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']
    # head_params = ['coref', 'mention', 'antecedent']
    #
    # model_decay = [p for n, p in model.named_parameters() if
    #                not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    # model_no_decay = [p for n, p in model.named_parameters() if
    #                   not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
    # head_decay = [p for n, p in model.named_parameters() if
    #               any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    # head_no_decay = [p for n, p in model.named_parameters() if
    #                  any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
    #
    # head_learning_rate = args.head_learning_rate if args.head_learning_rate else args.learning_rate
    # optimizer_grouped_parameters = [
    #     {'params': model_decay, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': model_no_decay, 'lr': args.learning_rate, 'weight_decay': 0.0},
    #     {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': head_no_decay, 'lr': head_learning_rate, 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters,
    #                   lr=args.learning_rate,
    #                   betas=(args.adam_beta1, args.adam_beta2),
    #                   eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
    #
    # loaded_saved_optimizer = False
    # # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #         os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    #     loaded_saved_optimizer = True
    #
    # if args.amp:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #
    # # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #
    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)
    #
    # # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)
    #
    # global_step = 0
    # if os.path.exists(args.model_name_or_path) and 'checkpoint' in args.model_name_or_path:
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         global_step = int(checkpoint_suffix)
    #
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from global step %d", global_step)
    #         if not loaded_saved_optimizer:
    #             logger.warning("Training is continued from checkpoint, but didn't load optimizer and scheduler")
    #     except ValueError:
    #         logger.info("  Starting fine-tuning.")
    # tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    # set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    #
    # # If nonfreeze_params is not empty, keep all params that are
    # # not in nonfreeze_params fixed.
    # if args.nonfreeze_params:
    #     names = []
    #     for name, param in model.named_parameters():
    #         freeze = True
    #         for nonfreeze_p in args.nonfreeze_params.split(','):
    #             if nonfreeze_p in name:
    #                 freeze = False
    #
    #         if freeze:
    #             param.requires_grad = False
    #         else:
    #             names.append(name)
    #
    #     print('nonfreezing layers: {}'.format(names))
    #
    # train_iterator = trange(
    #     0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    # )
    # # Added here for reproducibility
    # set_seed(args)
    # best_f1 = -1
    # best_global_step = -1
    # for _ in train_iterator:
    #     epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    #     for step, batch in enumerate(epoch_iterator):
    #         batch = tuple(tensor.to(args.device) for tensor in batch)
    #         input_ids, attention_mask, gold_clusters = batch
    #         model.train()
    #
    #         outputs = model(input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         gold_clusters=gold_clusters,
    #                         return_all_outputs=False)
    #         loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
    #         losses = outputs[-1]
    #
    #         if args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #             losses = {key: val.mean() for key, val in losses.items()}
    #         if args.gradient_accumulation_steps > 1:
    #             loss = loss / args.gradient_accumulation_steps
    #
    #         if args.amp:
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss.backward()
    #
    #         tr_loss += loss.item()
    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             optimizer.step()
    #             scheduler.step()  # Update learning rate schedule
    #             model.zero_grad()
    #             global_step += 1
    #
    #             # Log metrics
    #             if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
    #                 logger.info(f"\nloss step {global_step}: {(tr_loss - logging_loss) / args.logging_steps}")
    #                 tb_writer.add_scalar('Training_Loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
    #                 for key, value in losses.items():
    #                     logger.info(f"\n{key}: {value}")
    #
    #                 logging_loss = tr_loss
    #
    #             if args.local_rank in [-1, 0] and args.do_eval and args.eval_steps > 0 and global_step % args.eval_steps == 0:
    #                 results = evaluator.evaluate(model, prefix=f'step_{global_step}', tb_writer=tb_writer, global_step=global_step)
    #                 f1 = results["f1"]
    #                 if f1 > best_f1:
    #                     best_f1 = f1
    #                     best_global_step = global_step
    #                     # Save model checkpoint
    #                     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    #                     if not os.path.exists(output_dir):
    #                         os.makedirs(output_dir)
    #                     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #                     model_to_save.save_pretrained(output_dir)
    #                     tokenizer.save_pretrained(output_dir)
    #
    #                     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    #                     logger.info("Saving model checkpoint to %s", output_dir)
    #
    #                     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    #                     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    #                     logger.info("Saving optimizer and scheduler states to %s", output_dir)
    #                 logger.info(f"best f1 is {best_f1} on global step {best_global_step}")
    #             if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and \
    #                     (not args.save_if_best or (best_global_step == global_step)):
    #                 # Save model checkpoint
    #                 output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    #                 if not os.path.exists(output_dir):
    #                     os.makedirs(output_dir)
    #                 model_to_save = model.module if hasattr(model,
    #                                                         'module') else model  # Take care of distributed/parallel training
    #                 model_to_save.save_pretrained(output_dir)
    #                 tokenizer.save_pretrained(output_dir)
    #
    #                 torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    #                 logger.info("Saving model checkpoint to %s", output_dir)
    #
    #                 torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    #                 torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    #                 logger.info("Saving optimizer and scheduler states to %s", output_dir)
    #
    #     if 0 < t_total < global_step:
    #         train_iterator.close()
    #         break
    #
    # with open(os.path.join(args.output_dir, f"best_f1.json"), "w") as f:
    #     json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

    tb_writer.close()
    return global_step, tr_loss / global_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
