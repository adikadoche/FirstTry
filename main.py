# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import logging
from cli import parse_args
import torch
import wandb #TODO

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, LongformerModel

from eval import Evaluator
# from modeling import Adi
from data import get_dataset


logger = logging.getLogger(__name__)


def main():
    print("FML")
    args = parse_args()
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config_class = LongformerConfig
    base_model_prefix = "longformer"
    train_dataset = get_dataset(args, tokenizer, evaluate=False)
    model = LongformerModel.from_pretrained(args.model_name_or_path,
                                config=config,
                                cache_dir=args.cache_dir)

    model.to(args.device)

    evaluator = Evaluator(args, tokenizer)

    global_step, tr_loss = train(args, train_dataset, model, tokenizer, evaluator)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
