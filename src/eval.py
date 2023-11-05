import torch
import sys

from param import parse_args
from utils.misc import load_config, dump_config
from models.LogitScaleNetwork import LogitScaleNetwork
import models
from modelnet import make_modelnet40test, test_modelnet40
from objaverse_lvis import make_objaverse_lvis, test_objaverse_lvis, xyz_objaverse_lvis, \
rgb_objaverse_lvis, features_objaverse_lvis

from utils_func import load_model

# def test_dataset(config, model, dataset, device, text_proj=None):

cli_args, extras = parse_args(sys.argv[1:])
config = load_config("src/configs/test.yaml", cli_args = vars(cli_args), extra_args = extras)
device = torch.device("cuda:0")

# model_name = "OpenShape/openshape-pointbert-no-lvis"
# model_name = "OpenShape/openshape-pointbert-shapenet"
model_name = "OpenShape/openshape-pointbert-vitg14-rgb"

model = load_model(config, model_name=model_name)
model.to(device)


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

# test_modelnet40(model, config, modelnet40_loader, text_proj, device)
# test_objaverse_lvis(model, config, objaverse_lvis_loader, text_proj, device)
# xyz_objaverse_lvis(objaverse_lvis_loader, device)
rgb_objaverse_lvis(objaverse_lvis_loader, device)
features_objaverse_lvis(objaverse_lvis_loader, device)