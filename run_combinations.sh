#!/bin/bash

RUN_SCRITP='run_with_slurm.sh'
RUN_NAME='gold_slots_overfit'
TRAIN_FILE='/home/gamir/adiz/datasets/ontonotes/50sameoverfit/train.english.jsonlines'
DEV_FILE='/home/gamir/adiz/datasets/ontonotes/50sameoverfit/dev.english.jsonlines'
declare -a PARAMS=('--slots' '--cluster_block' '--use_gold_mentions')

PARAMS_LENGTH=${#PARAMS[@]}
for i in $(seq 0 1 $(echo "2^$PARAMS_LENGTH-1" | bc))
do
    # echo $i
    chosen_params=()
    for j in $(seq ${#PARAMS[@]} -1 0)
    do
        J_POWER=$(echo "${j}^2" | bc)
        if [ "$J_POWER" -lt "$i" ]
        then
            chosen_params+=(${PARAMS[$j]})
            echo "$i, $j, ${PARAMS[$j]}"
            i=$((i-J_POWER))
        fi
    done
    for value in "${chosen_params[@]}"
    do
        echo $value
    done
    echo "-----------------------------"
        # bash $RUN_SCRITP \
        #     $RUN_NAME train_slurm.sh \
        #     --output_dir /home/gamir/adiz/Code/runs/firsttry/output_dir/ \
        #     --cache_dir /home/gamir/adiz/Code/runs/firsttry/cache_dir/ \
        #     --max_eval_print 25 \
        #     --model_type longformer \
        #     --model_name_or_path allenai/longformer-base-4096 \
        #     --tokenizer_name allenai/longformer-base-4096 \
        #     --config_name allenai/longformer-base-4096 \
        #     --train_file $TRAIN_FILE \
        #     --predict_file $DEV_FILE \
        #     --do_train --eval all \
        #     --num_train_epochs 40 \
        #     --logging_steps 50 --save_steps -1 --eval_steps -1 --eval_epochs 1 --max_seq_length 4096 --gradient_accumulation_steps 1 \
        #     --max_total_seq_len 5000 --warmup_steps 5000 --weight_decay 0.01 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --save_epochs 1 \
        #     --num_queries 50 \
        #     --reduction sum \
        #     --input_type ontonotes \
        #     --is_cluster \
        #     --is_encoding \
        #     --is_max \
        #     --slots \
        #     --use_gold_mentions
    # done
done