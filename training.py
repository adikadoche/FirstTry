import shutil
import os
import logging
import random
import numpy as np
import wandb
import torch
from eval import evaluate, report_eval
from tqdm import tqdm, trange
import time, datetime
from misc import save_on_master, is_main_process
from utils import tensor_and_remove_empty, create_gold_matrix, calc_predicted_clusters, create_junk_gold_mentions, try_measure_len
from optimization import WarmupLinearSchedule, WarmupExponentialSchedule
import itertools
from metrics import CorefEvaluator
from utils import load_from_checkpoint, save_checkpoint
from consts import TOKENS_PAD, SPEAKER_PAD

from transformers import AdamW, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    epoch_iterator, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    args, evaluator, skip_steps, recent_grad_norms, recent_cluster_logits,
        recent_coref_logits, recent_losses, recent_losses_parts, recent_logits_sums, global_step, lr_scheduler, eval_loader, eval_dataset, threshold):
    input_ids_pads = torch.ones(1, args.max_segment_len, dtype=torch.int, device=args.device) * TOKENS_PAD
    speaker_ids_pads = torch.ones(1, args.max_segment_len, args.max_num_speakers, dtype=torch.int, device=args.device) * SPEAKER_PAD
    mask_pads = torch.zeros(1, args.max_segment_len, dtype=torch.int, device=args.device)
    for step, batch in enumerate(epoch_iterator):
        if skip_steps > 0:
            skip_steps -= 1
            continue

        model.train()

        sum_text_len = [sum(tl) for tl in batch['text_len']]
        gold_clusters = batch['clusters']

        gold_mentions_list = None
        if args.use_gold_mentions:
            # gold_mentions = []
            # if len(gold_clusters) > 0:  #TODO: create junk clusters even if 0 gold clusters
            gold_mentions_list = [list(set([tuple(m) for c in gc for m in c])) for gc in gold_clusters]
            if args.add_junk:
                gold_mentions_list, gold_mentions_vector = create_junk_gold_mentions(gold_mentions_list, sum_text_len, args.device)
            else:
                gold_mentions_vector = [torch.ones(len(gm), dtype=torch.float, device=args.device) for gm in gold_mentions_list]

        input_ids, input_mask, sum_text_len, gold_mentions, num_mentions, speaker_ids, genre = tensor_and_remove_empty(batch, gold_mentions_list, args, input_ids_pads, mask_pads, speaker_ids_pads)
        if len(input_ids) == 0:
            continue

        gold_matrix = create_gold_matrix(args.device, sum_text_len, args.num_queries, gold_clusters, gold_mentions_list)

        # orig_input_dim = input_ids.shape
        # input_ids = torch.reshape(input_ids, (1, -1))
        # input_mask = torch.reshape(input_mask, (1, -1))
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, sum_text_len, input_mask, gold_mentions, num_mentions)
                cluster_logits, coref_logits = outputs['cluster_logits'], outputs['coref_logits']

                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(),
                                                            threshold, gold_mentions_list, args.is_max or args.detr)
                evaluator.update(predicted_clusters, gold_clusters)
                loss = criterion(outputs, gold_matrix)
        else:
            outputs = model(input_ids, sum_text_len, input_mask, gold_mentions, num_mentions)
            cluster_logits, coref_logits, mention_logits = outputs['cluster_logits'], outputs['coref_logits'], outputs['mention_logits']

            if args.add_junk:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), mention_logits.cpu().detach(),
                                                            threshold, gold_mentions_list, args.is_max or args.detr)
            else:
                predicted_clusters = calc_predicted_clusters(cluster_logits.cpu().detach(), coref_logits.cpu().detach(), [],
                                                            threshold, gold_mentions_list, args.is_max or args.detr)
            evaluator.update(predicted_clusters, gold_clusters)
            loss, loss_parts = criterion(outputs, {'clusters':gold_matrix, 'mentions':gold_mentions_vector})

        # recent_cluster_logits.append(cluster_logits.detach().cpu().numpy())
        # recent_coref_logits.append(coref_logits.detach().cpu().numpy().flatten())
        # recent_logits_sums.append(coref_logits.detach().sum(1).flatten().cpu().numpy())

        # TODO handle NaNs and +-infs

        if args.n_gpu > 1 or args.train_batch_size > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # if args.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        #     total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        # else:
        # print("before")
        # print(model.IO_score[0].weight)
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # recent_grad_norms.append(total_norm.item())
        recent_losses.append(loss.item())
        for key in loss_parts.keys():
            if key in recent_losses_parts.keys() and len(recent_losses_parts[key]) > 0:
                recent_losses_parts[key] += loss_parts[key]
            else:
                recent_losses_parts[key] = loss_parts[key]
        epoch_iterator.set_postfix({'loss': loss.item()})

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if args.lr_drop_interval == 'step':
                lr_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += args.train_batch_size

            if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, threshold)
                threshold = results['threshold']

            # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            #     save_checkpoint(args, global_step, threshold, model, optimizer)

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # logits = np.concatenate(recent_cluster_logits)
                # wandb.log({'all_cluster_logits': logits}, step=global_step)
                # wandb.log({'current_cluster_logits': cluster_logits.detach().cpu().numpy()}, step=global_step)
                # recent_cluster_logits.clear()
                # logits = np.concatenate(recent_coref_logits)
                # wandb.log({'all_recent_coref_logits': logits}, step=global_step)
                # wandb.log({'current_recent_coref_logits': coref_logits.detach().cpu().numpy()}, step=global_step)
                # recent_coref_logits.clear()
                # wandb.log({'grad_total_norm': np.mean(recent_grad_norms)}, step=global_step)
                # recent_grad_norms.clear()
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=global_step)
                wandb.log({'lr_bert': optimizer.param_groups[1]['lr']}, step=global_step)
                wandb.log({'loss': np.mean(recent_losses)}, step=global_step)
                for key in recent_losses_parts.keys():
                    wandb.log({key: np.mean(recent_losses_parts[key])}, step=global_step)
                recent_losses.clear()
                recent_losses_parts.clear()
                # wandb.log({'coref_logits_sum_over_clusters': np.concatenate(recent_logits_sums)}, step=global_step)
                # recent_logits_sums.clear()

        # print("after")
        # print(model.IO_score[0].weight)
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    return global_step, threshold


def train(args, model, criterion, train_loader, eval_loader, eval_dataset):
    """ Train the model """
    # output_dir = Path(args.output_dir)

    logger.info("Training/evaluation parameters %s", args)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone, #TODO: learn how to freeze backbone
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr * (1 + (args.train_batch_size- 1)/40.0),
                                  weight_decay=args.weight_decay)
    
    if args.resume_from:
        logger.info("Loading from checkpoint {}".format(args.resume_from))
        loaded_args = load_from_checkpoint(model, args.resume_from, args, args.device, optimizer)
        args.resume_global_step = int(loaded_args['global_step'])
        if not args.do_train:
            return args.resume_global_step
                                 

    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # loaded_saved_optimizer = False
    # Check if saved optimizer or lr_scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #         os.path.join(args.model_name_or_path, "lr_scheduler.pt")
    # ):
    #     # Load in optimizer and lr_scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     lr_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "lr_scheduler.pt")))
    #     loaded_saved_optimizer = True

    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print("FML")

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    
    if args.max_steps > 0:
        args.t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
    else:
        args.t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps / args.train_batch_size))
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int((args.warmup_steps/args.train_batch_size) * (1 + (args.train_batch_size - 1) / 5)),
                                        t_total=args.t_total)  # ConstantLRSchedule(optimizer)
    # lr_scheduler = WarmupExponentialSchedule(optimizer, warmup_steps=int(args.warmup_steps / args.train_batch_size),
    #                                     gamma=0.99998)  # ConstantLRSchedule(optimizer)
    
    if args.train_batch_size > 1:
        args.eval_steps = -1 if args.eval_steps == -1 else max(1, int(round(args.eval_steps / args.train_batch_size)))
        args.save_steps = -1 if args.save_steps == -1 else max(1, int(round(args.save_steps / args.train_batch_size)))
        args.logging_steps = -1 if args.logging_steps == -1 else max(1, int(round(args.logging_steps / args.train_batch_size)))


    global_step = 0 if not args.resume_from else args.resume_global_step
    if args.local_rank in [-1, 0]:
        purge_step = None if not args.resume_from else args.resume_global_step

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num steps per epoch = %d", try_measure_len(train_loader))
    logger.info("  Num Epochs = %d", args.num_train_epochs if args.num_train_epochs is not None else -1)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)


    model.zero_grad()
    recent_grad_norms = []
    recent_cluster_logits = []
    recent_coref_logits = []
    recent_losses = []
    recent_losses_parts = {}
    recent_logits_sums = []
    train_iterator = itertools.count() if args.num_train_epochs is None else range(int(args.num_train_epochs))
    train_iterator = tqdm(train_iterator, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    skip_steps = args.skip_steps
    threshold = 0.5 # starting threshold, later fixed by eval

    best_f1 = 0
    best_f1_global_step = -1
    start_time = time.time()
    for epoch in train_iterator:
        # if epoch > len(train_iterator) / 2:
        #     args.add_junk = True
        evaluator = CorefEvaluator()
        epoch_iterator = tqdm(train_loader, desc="Iteration in Epoch {}".format(epoch), disable=args.local_rank not in [-1, 0], leave=False)
        global_step, threshold = train_one_epoch(   #TODO: do I need to let the threshold return to 0.5 every time? or is it correct to update it?
            model, criterion, epoch_iterator, optimizer, scaler, args, evaluator, skip_steps, recent_grad_norms,
            recent_cluster_logits, recent_coref_logits, recent_losses, recent_losses_parts, recent_logits_sums, global_step,
            lr_scheduler, eval_loader, eval_dataset, threshold)

        t_p, t_r, t_f1 = evaluator.get_prf()
        if args.local_rank in [-1, 0]:
            wandb.log({'Train Precision':t_p}, step=global_step)
            wandb.log({'Train Recall': t_r}, step=global_step)
            wandb.log({'Train F1': t_f1}, step=global_step)
            logger.info(f'Train step {global_step} f1 {t_f1}, precision {t_p} , recall {t_r}')

        if args.lr_drop_interval == 'epoch':
            lr_scheduler.step()  # Update learning rate schedule

        if args.local_rank in [-1, 0]:
            if args.eval_epochs > 0 and (epoch + 1) % args.eval_epochs == 0 or \
                    epoch + 1 == args.num_train_epochs and (args.eval_epochs > 0 or args.eval_steps > 0):
                results = report_eval(args, eval_loader, eval_dataset, global_step, model, criterion, threshold)
                threshold = results['threshold']
                f1 = results['avg_f1']
            logger.info(f'Train step {global_step} f1 {t_f1}, precision {t_p} , recall {t_r}')

            if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0 or epoch + 1 == args.num_train_epochs:
                if f1 > best_f1:
                    prev_best_f1 = best_f1
                    prev_best_f1_global_step = best_f1_global_step
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    save_checkpoint(args, global_step, threshold, model, optimizer, output_dir)
                    print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_global_step}')
                    best_f1 = f1
                    best_f1_global_step = global_step
                    print(f'saved checkpoint with f1 {best_f1} in step {best_f1_global_step} to {output_dir}')
                    if prev_best_f1_global_step > -1:
                        path_to_remove = os.path.join(args.output_dir, 'checkpoint-{}'.format(prev_best_f1_global_step))
                        shutil.rmtree(path_to_remove)
                        print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')


        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step

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
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
    #
    # loaded_saved_optimizer = False
    # # Check if saved optimizer or lr_scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #         os.path.join(args.model_name_or_path, "lr_scheduler.pt")
    # ):
    #     # Load in optimizer and lr_scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     lr_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "lr_scheduler.pt")))
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
    #             logger.warning("Training is continued from checkpoint, but didn't load optimizer and lr_scheduler")
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
    #             lr_scheduler.step()  # Update learning rate schedule
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
    #                     torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "lr_scheduler.pt"))
    #                     logger.info("Saving optimizer and lr_scheduler states to %s", output_dir)
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
    #                 torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "lr_scheduler.pt"))
    #                 logger.info("Saving optimizer and lr_scheduler states to %s", output_dir)
    #
    #     if 0 < t_total < global_step:
    #         train_iterator.close()
    #         break
    #
    # with open(os.path.join(args.output_dir, f"best_f1.json"), "w") as f:
    #     json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)
    #
    # tb_writer.close()
    # return global_step, tr_loss / global_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
