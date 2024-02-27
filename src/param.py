import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/train.yaml",
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default="try",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./exp",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train a model."
    )
    parser.add_argument(
        "--load_from_online",
        action="store_true",
        help="load model from huggingface"
    )
    parser.add_argument(
        "--resume", 
        default=None, 
        help="path to the weights to be resumed"
    )
    parser.add_argument(
        "--autoresume",
        action="store_true",
        help="auto back-off on failure"
    )
    parser.add_argument(
        "--ngpu", 
        default=1, 
        type=int,
        help="number of gpu used"
    )
    parser.add_argument(
        "--text_model",
        default="clip",
        type=str,
        help="text model used to generate text features"
    )
    parser.add_argument(
        "--wandb_key",
        # default="3051f76ad3503c10871179fb154be79d5e561ccc",
        default=None,
        type=str,
        help="wandb login key"
    )
    parser.add_argument(
        "--cache_dir",
        default="/projectnb/ivc-ml/harshk/.cache",
        type=str,
        help="cache directory"
    )
    parser.add_argument(
        "--export_results",
        action="store_true",
        help="export results to .pt file"
    )
    parser.add_argument(
        "--export_result_dir",
        default=None,
        type=str,
        help="directory to export results"
    )
    parser.add_argument(
        "--default_batch_size",
        action="store_true",
        help="use default batch size as whole dataset"
    )
    args, extras = parser.parse_known_args()
    return args, extras
