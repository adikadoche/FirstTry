import argparse

def parse_args():
    parser = argparse.ArgumentParser()

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



    args = parser.parse_args()
    return args
