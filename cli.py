import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default="longformer",
        type=str,
        # required=True,
        # help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="allenai/longformer-base-4096",
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        default="allenai/longformer-base-4096",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="allenai/longformer-base-4096",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")


    args = parser.parse_args()
    return args
