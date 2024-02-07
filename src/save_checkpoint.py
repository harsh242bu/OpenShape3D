import torch
import sys
from collections import OrderedDict
import re

import models
from param import parse_args
from utils.misc import load_config, dump_config
from utils_func import load_model, load_model_from_path, load_model_no_classification
from utils_func import hf_hub_download

cli_args, extras = parse_args(sys.argv[1:])
# config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)

model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
model_dict = OrderedDict()
pattern = re.compile('module.')
for k,v in checkpoint['state_dict'].items():
    if re.search("module", k):
        model_dict[re.sub(pattern, '', k)] = v

# checkpoint = torch.load(path)
device = torch.device("cuda")
model = models.make(config).to(device)
model.load_state_dict(model_dict)

checkpoint['state_dict'] = model_dict

torch.save(checkpoint, "pretrained/openshape-pointbert-vitg14-rgb/checkpoint.pt")

# model.load_state_dict(checkpoint['state_dict'])

# if config.training.use_text_proj:
#     text_proj.load_state_dict(checkpoint['text_proj'])
# if config.training.use_image_proj:
#     image_proj.load_state_dict(checkpoint['image_proj'])
# logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
# optimizer.load_state_dict(checkpoint['optimizer'])
# if config.training.use_openclip_optimizer_scheduler == False:
#     scheduler.load_state_dict(checkpoint['scheduler'])
# epoch = checkpoint['epoch']
# step = checkpoint['step']
# best_img_contras_acc = checkpoint['best_img_contras_acc']
# best_text_contras_acc = checkpoint['best_text_contras_acc']
# best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
# best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
# best_lvis_acc = checkpoint['best_lvis_acc']

print("Model loaded from checkpoint")
# print("Model: ", model)