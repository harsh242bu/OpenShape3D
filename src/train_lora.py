import torch
import sys
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora
from minlora import name_is_lora, remove_lora, load_multiple_lora, select_lora

from param import parse_args
from utils.misc import load_config, dump_config
from utils_func import load_model, load_model_from_path, load_model_no_classification

cli_args, extras = parse_args(sys.argv[1:])
# config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device: ", device)

torch.manual_seed(2024)

model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
base_model = load_model(config, model_name=model_name)

add_lora(base_model)

parameters = [{
    "params": list(get_lora_params(base_model))
}]
optimizer = torch.optim.Adam(parameters, lr=0.001)

