bash run_with_slurm.sh old_DETR_improv train_slurm.sh --output_dir /home/gamir/adiz/Code/runs/firsttry/output_dir/ --cache_dir /home/gamir/adiz/Code/runs/firsttry/cache_dir/ --max_eval_print 25 --model_type longformer --model_name_or_path allenai/longformer-large-4096 --tokenizer_name allenai/longformer-large-4096 --config_name allenai/longformer-large-4096 --train_file /home/gamir/datasets/e2e-coref/train.english.jsonlines --predict_file /home/gamir/datasets/e2e-coref/dev.english.jsonlines --do_train --eval all --num_train_epochs 40 --logging_steps 50 --save_steps -1 --eval_steps -1 --eval_epochs 1 --max_seq_length 4096 --gradient_accumulation_steps 1 --max_total_seq_len 5000 --warmup_steps 5000 --weight_decay 0.01 --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --save_epochs 1 --num_queries 100 --use_gold_mentions 