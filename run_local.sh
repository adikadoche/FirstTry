#! /bin/bash

export JOB_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
export GIT_HASH="$(git rev-parse HEAD)"
LOG_DIR="slurm_logs"

if [[ ${JOB_NAME} == *".sh"* ]]; then
  echo "woops! your first argument contains '.sh', did you forget to specify a slurm job name?"
  exit 1
fi

mkdir --parents ${LOG_DIR}

DATE=$(date +"%Y-%m-%d-%H-%M-%S-%N")
DATE=${DATE::23}
LOG_PATH=${LOG_DIR}/${DATE}_${JOB_NAME}_slurm_log.txt
echo ""
echo $LOG_PATH

PYTHONUNBUFFERED=1 nohup /home/gamir/adiz/miniconda3/envs/torchGPU/bin/python -u \
  main.py \
  --output_dir /home/gamir/adiz/Code/runs/firsttry/output_dir/ --cache_dir /home/gamir/adiz/Code/runs/firsttry/cache_dir/ --max_eval_print 5 \
  --model_type longformer --bert_model roberta-large \
  --train_file /home/gamir/adiz/datasets/wl-ontonotes/english_train_head.jsonlines --predict_file /home/gamir/adiz/datasets/wl-ontonotes/english_dev_head.jsonlines --do_train --eval all \
  --num_train_epochs 100 --logging_steps 50 --save_steps -1 --eval_steps -1 --eval_epochs 1 --max_seq_length 4096 --gradient_accumulation_steps 1 \
  --max_total_seq_len 5000 --warmup_steps 5000 --weight_decay 0.01 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --save_epochs 1 --num_queries 150 \
  --slots --use_topk_mentions --max_grad_norm 1.0 --cluster_block --num_junk_queries 200 \
  --loss max --topk_lambda 0.4 --lr 1e-5 --lr_backbone 1e-6 |  tee ${LOG_PATH}  &

