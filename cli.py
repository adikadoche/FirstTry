import argparse

MODEL_TYPES = ['longformer']


def parse_args():
    parser = argparse.ArgumentParser()

    # Temporal
    parser.add_argument(
        "--is_softmax",
        action="store_true",
    )
    # Required parameters
    parser.add_argument(
        "--model_type",
        default="longformer",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/longformer-base-4096",
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--resume_from", default='', type=str)
    parser.add_argument(
        "--train_file_cache",
        default=None,
        type=str,
        required=True,
        help="The output directory where the datasets will be written and read from.",
    )
    parser.add_argument(
        "--predict_file_cache",
        default=None,
        type=str,
        required=True,
        help="The output directory where the datasets will be written and read from.",
    )

    # Other parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="allenai/longformer-base-4096", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument("--tokenizer_name",
                        default="allenai/longformer-base-4096",
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=-1, type=int)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--eval", type=str, choices=['no', 'specific', 'all', 'vanilla'], default='no')
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--nonfreeze_params", default=None, type=str,
                        help="named parameters to update while training (separated by ,). The rest will kept frozen. If None or empty - train all")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--head_learning_rate", default=3e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.3, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")
    parser.add_argument('--eval_epochs', type=int, default=1)
    parser.add_argument('--eval_sleep', type=int, default=10)
    parser.add_argument('--max_eval_print', type=int, default=10)
    parser.add_argument('--eval_skip_until', type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(   #TODO: make amp work
        "--amp",
        action="store_true",
        help="Whether to use automatic mixed precision instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--max_span_length", type=int, required=False, default=30)
    parser.add_argument("--top_lambda", type=float, default=0.4)

    parser.add_argument("--max_total_seq_len", type=int, default=3500)
    parser.add_argument("--experiment_name", type=str, default=None)


    parser.add_argument("--normalise_loss", action="store_true")

    parser.add_argument("--ffnn_size", type=int, default=3072)

    parser.add_argument("--save_if_best", action="store_true")
    parser.add_argument("--batch_size_1", action="store_true")
    parser.add_argument("--tensorboard_dir", type=str, required=True)

    parser.add_argument("--conll_path_for_eval", type=str, default=None)


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--multiclass_ce', action='store_true')
    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=1, type=float)
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    # parser.add_argument('--bbox_loss_coef', default=5, type=float)
    # parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--cost_is_cluster', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--cost_coref', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--max_training_sentences', default=3, type=int)
    parser.add_argument('--max_num_speakers', default=20, type=int)
    parser.add_argument('--max_segment_len', default=512, type=int)
    parser.add_argument('--limit_trainset', default=-1, type=int)
    parser.add_argument('--eval_as_train', action='store_true')
    parser.add_argument('--use_gold_mentions', action='store_true')
    parser.add_argument('--softmax_coref', action='store_true')
    parser.add_argument('--random_queries', action='store_true')
    parser.add_argument('--single_distribution_queries', action='store_true')
    parser.add_argument('--fc_coref_head', action='store_true')
    parser.add_argument('--pairwise_score', action='store_true')
    parser.add_argument('--sum_attn', action='store_true')
    parser.add_argument('--attn_softmax_clusters', action='store_true')

    # Batch
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--num_workers', type=int, default=0)

    # Epochs
    parser.add_argument("--num_train_epochs", default=300.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=5000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--skip_steps", default=0, type=int)


    parser.add_argument('--lr_backbone', default=1e-5, type=float) #TODO: remove
    parser.add_argument('--lr', default=1e-4, type=float) #TODO:?
    parser.add_argument('--lr_drop', default=200, type=int) #TODO:?
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                    help="Max gradient norm.")
    parser.add_argument('--lr_drop_interval', default="step", type=str, choices=['epoch, step'])


    args = parser.parse_args()
    return args
