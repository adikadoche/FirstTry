import os
from itertools import combinations

# params = [(("--slots", True), ["slots", "DETR"]),
# (("--cluster_block", True), ["block"]),
# (("--use_gold_mentions", True), ["gold"]),
# (("--add_junk", True), ["junk"]),
# (("--min_cluster_size 1", True), ["no1"]),
# (("--input_type", False), ["ontonotes", "sequences_9950"]),
# (("--load_backbone", False), ["best", "latest", "no"])]
params = [(("--slots", True), ["slots", "DETR"]),
(("--cluster_block", True), ["block"]),
# (("--min_cluster_size 1", True), ["no1"]),
# (("--use_gold_mentions", True), ["gold"]),
# (("--add_junk", True), ["junk"]),
# (("--input_type", False), ["ontonotes", "sequences_9950"]),
(("--load_backbone", False), ["best", "latest"])
]

x=0
for i in range(0, len(params)+1):
    for c in list(combinations(range(len(params)), i)):
        name = []
        flags = []
        for ind in range(len(params)):
            if ind in c:
                name.append(params[ind][1][0])
                flags.append(params[ind][0][0])
                if not params[ind][0][1]:
                    flags[-1] += ' ' + params[ind][1][0]
            elif len(params[ind][1]) > 1:
                name.append(params[ind][1][1])
                if not params[ind][0][1]:
                    flags.append(params[ind][0][0] + ' ' + params[ind][1][1])
        if "--use_gold_mentions" not in flags and "--add_junk" in flags:
            continue
        num_iter = 15 if "sequences_9950" in name else 40
        name = '_'.join(name) + '_squences'
        flags = " ".join(flags)
        # print(name)
        # print(flags)
        x+=1
        os.system(f"bash run_stud_with_slurm.sh {name} train_slurm.sh --output_dir /home/gamir/adiz/Code/runs/firsttry/output_dir/ --cache_dir /home/gamir/adiz/Code/runs/firsttry/cache_dir/ \
            --max_eval_print 25 --model_type longformer --model_name_or_path allenai/longformer-base-4096 --tokenizer_name allenai/longformer-base-4096 \
                --config_name allenai/longformer-base-4096 --train_file /home/gamir/datasets/e2e-coref/train.english.jsonlines \
                    --predict_file /home/gamir/datasets/e2e-coref/dev.english.jsonlines --do_train --eval all --num_train_epochs {num_iter} --logging_steps 50 --save_steps -1 --eval_steps -1 \
                        --eval_epochs 1 --max_seq_length 4096 --gradient_accumulation_steps 1 --max_total_seq_len 5000 --warmup_steps 5000 --weight_decay 0.01 \
                            --per_gpu_eval_batch_size 1 --per_gpu_train_batch_size 1 --save_epochs 1 --num_queries 70 --reduction sum --is_cluster \
                                --input_type sequences_9950 {flags}")
print(x)
