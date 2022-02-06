#! /bin/bash

export JOB_NAME=$1
export GIT_HASH="$(git rev-parse HEAD)"
LOG_DIR="slurm_logs"

export CUDA_VISIBLE_DEVICES="1"
# export CUDA_LAUNCH_BLOCKING="1"

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

PYTHONUNBUFFERED=1 /home/gamir/adiz/miniconda3/envs/torchGPU/bin/python -u \
  main.py \
  --output_dir /home/gamir/adiz/Code/runs/firsttry/output_dir/ --cache_dir /home/gamir/adiz/Code/runs/firsttry/cache_dir/ --max_eval_print 10 \
  --model_type longformer --model_name_or_path allenai/longformer-large-4096 --tokenizer_name allenai/longformer-large-4096 --config_name allenai/longformer-large-4096 \
  --train_file /home/gamir/datasets/e2e-coref/train.english.jsonlines --predict_file /home/gamir/datasets/e2e-coref/dev.english.jsonlines --do_train --eval all \
  --num_train_epochs 60 --logging_steps 50 --save_steps -1 --eval_steps -1 --eval_epochs 1 --max_seq_length 4096 --gradient_accumulation_steps 1 \
  --max_total_seq_len 5000 --warmup_steps 8000 --weight_decay 0.01 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --save_epochs 1 \
   --num_queries 100 --num_junk_queries 150 --cluster_block --slots --use_topk_mentions --topk_pre --max_grad_norm 1.0 --topk_lambda 0.2 \
   --reset_optim --resume_from /home/gamir/adiz/Code/runs/firsttry/output_dir/01_25_2022_16_52_29_bestfreeze1prenosep_topjunkkslots_block/checkpoint-112040 \
   --lr 0.00001 --lr_backbone 0.000001 \
  |  tee ${LOG_PATH}  &

