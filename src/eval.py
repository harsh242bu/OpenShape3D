import torch
import sys
import os 

from param import parse_args
from utils.misc import load_config, dump_config
from models.LogitScaleNetwork import LogitScaleNetwork
import models
from modelnet import make_modelnet40test, test_modelnet40
from objaverse_lvis import make_objaverse_lvis, test_objaverse_lvis, xyz_objaverse_lvis, \
rgb_objaverse_lvis, features_objaverse_lvis
import data

from utils_func import load_model, load_model_from_path 
# from utils_func import load_lora_model_from_path

# def test_dataset(config, model, dataset, device, text_proj=None):

torch.manual_seed(2024)

cli_args, extras = parse_args(sys.argv[1:])
# config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device: ", device)

# model = load_model(config, model_name=model_name)
# model.to(device)

if config.resume is not None:
    model = load_model_from_path(config, config.resume, device)
    # model = load_lora_model_from_path(config, config.resume, device)
else:
    # model_name = "OpenShape/openshape-pointbert-no-lvis"
    # model_name = "OpenShape/openshape-pointbert-shapenet"
    model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
    
    model = load_model(config, model_name=model_name)
    model.to(device)


# print("model: ", model)
# quit()

# config = load_config(config)
# if config.autoresume:
#     config.trial_name = config.get('trial_name') + "@autoresume"
# else:
#     config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
# config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
# config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

# config.device = 'cuda:{0}'.format(rank) 

# model = models.make(config).to(device)
# if rank == 0:
#     total_params = sum(p.numel() for p in model.parameters())
#     logging.info(model)
#     logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))

# torch.cuda.set_device(rank)
# model.cuda(rank)
# model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
# logging.info("Using SyncBatchNorm")

logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(device)
image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)

# modelnet40_loader = make_modelnet40test(config)
objaverse_lvis_loader = make_objaverse_lvis(config)
# scanobjectnn_loader = make_scanobjectnntest(config)

phase = "test"
# finetune_test_loader = data.make(config, phase, None, None)

print("objaverse_lvis total batches: ", len(objaverse_lvis_loader))
print("objaverse_lvis dataset size: ", len(objaverse_lvis_loader.dataset))
# print("finetune_test_loader batches: ", len(finetune_test_loader))
# print("finetune_test_loader dataset size: ", len(finetune_test_loader.dataset))


# test_modelnet40(model, config, modelnet40_loader, text_proj, device)
objaverse_dict = test_objaverse_lvis(model, config, objaverse_lvis_loader, text_proj, device)
# objaverse_dict = test_objaverse_lvis(model, config, finetune_test_loader, text_proj, device)

if config.export_result_dir and not os.path.exists(config.export_result_dir):
    os.makedirs(config.export_result_dir)

if config.export_result_dir is not None:
    # Resume is for the finetuned model
    if config.resume is not None:
        suffix = config.resume.split("/")[-1].split(".")[0]
        phase = "ov_test_all"
        torch.save(objaverse_dict["labels_all"], os.path.join(config.export_result_dir, f"ov_labels_all_finetune_{phase}_{suffix}.pt"))
        torch.save(objaverse_dict["logits_all"], os.path.join(config.export_result_dir, f"ov_logits_all_finetune_{phase}_{suffix}.pt"))

    # Else condition is for the baseline pretrained model
    else:
        torch.save(objaverse_dict["labels_all"], os.path.join(config.export_result_dir, f"ov_labels_all_finetune_{phase}.pt"))
        torch.save(objaverse_dict["logits_all"], os.path.join(config.export_result_dir, f"ov_logits_all_finetune_{phase}.pt"))

# xyz_objaverse_lvis(objaverse_lvis_loader, device)
# rgb_objaverse_lvis(objaverse_lvis_loader, device)
# features_objaverse_lvis(objaverse_lvis_loader, device)
        