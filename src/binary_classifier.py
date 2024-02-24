import sys
import os
import logging
import shutil
import data
import models
# import MinkowskiEngine as ME
import torch
import wandb
from torch import nn
from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config    
from utils.logger import setup_logging
from utils.scheduler import cosine_lr
from train_binary import Trainer
from models.LogitScaleNetwork import LogitScaleNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from utils_func import load_model
from finetune_loader import FinetuneLoader

target_cat_pair = "desk_monitor"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_cache(config):
    if config.cache_dir is not None:
        print("Setting cache: ", config.cache_dir)
        os.environ['TORCH_HOME'] = os.path.join(config.cache_dir, 'torch')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(config.cache_dir, 'huggingface', 'transformers')
        os.environ['HF_DATASETS_CACHE'] = os.path.join(config.cache_dir, 'huggingface', 'datasets')
        os.environ['HF_HOME'] = os.path.join(config.cache_dir, 'huggingface', 'home')

def cleanup():
    dist.destroy_process_group()

class BinaryClassifier(torch.nn.Module):
    def __init__(self, base_model, middle_layer=256):
        super(BinaryClassifier, self).__init__()
        self.base_model = base_model
        self.fc1 = torch.nn.Linear(1156, middle_layer)
        self.fc2 = torch.nn.Linear(middle_layer, 1)

        # Freeze the layers
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, xyz, features):
        # print(f"input_tuple: {input_tuple}")
        # xyz, features = input_tuple
        # print("xyz: ", xyz.shape)
        # print("features: ", features.shape)
        x = self.base_model(xyz, features)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def main(rank, world_size, cli_args, extras):
    setup(rank, world_size)
    config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)

    setup_cache(config)
    logging.info("config: {}".format(config))

    config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        shutil.copytree("./src", config.code_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    config.device = 'cuda:{0}'.format(rank)

    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))
        if config.wandb_key is not None:
            wandb.login(key=config.wandb_key)
            wandb.init(project=config.project_name, name=config.trial_name, config=OmegaConf.to_object(config))
    
    model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
    base_model = load_model(config, model_name)
    model = BinaryClassifier(base_model).to(config.device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(model)
        logging.info("Network:{}, Number of parameters: {}".format(model_name, total_params))

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logging.info("Using SyncBatchNorm")

    train_loader = data.make(config, 'train', rank, world_size, target_cat_pair)

    if rank == 0:
        test_loader = data.make(config, 'test', rank, world_size, target_cat_pair)
    else:
        test_loader = None

    print("train_loader size: ", len(train_loader))
    print("train_loader dataset size: ", len(train_loader.dataset))

    print("test_loader size: ", len(test_loader))
    print("test_loader dataset size: ", len(test_loader.dataset))

    if rank == 0 and train_loader is not None:
        logging.info("Train iterations: {}".format(len(train_loader)))

    params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    # params = list(model.parameters())
    logging.info("Trainable parameters: {}".format(len(params)))
    print("Parameters: ", params)

    if config.training.use_openclip_optimizer_scheduler:
        optimizer = torch.optim.AdamW(
            params,
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
        )
        # scheduler = cosine_lr(optimizer, config.training.lr, config.training.warmup, len(train_loader) * 16)
    else:
        optimizer = torch.optim.AdamW(params, lr=config.training.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step, gamma=config.training.lr_decay_rate)
    
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(config, model, optimizer, None, criterion, train_loader, test_loader)
    trainer.train()

    if rank == 0 and config.wandb_key is not None:
        wandb.finish()
    cleanup()

# def eval(config):
#     setup_cache(config)
#     print("config: ", config)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device: ", device)

#     config.device = device



#     model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
#     base_model = load_model(config, model_name)
#     model = BinaryClassifier(base_model).to(device)

#     total_params = sum(p.numel() for p in model.parameters())
#     logging.info(model)
#     logging.info("Network:{}, Number of parameters: {}".format(model_name, total_params))

#     logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
#     image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
#     text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)

#     train_loader = FinetuneLoader(config, 'train', target_cat_pair)
#     print("train_loader size: ", len(train_loader))
#     print("train_loader dataset size: ", len(train_loader.dataset))

#     if train_loader is not None:
#         logging.info("Train iterations: {}".format(len(train_loader)))

#     params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + list(logit_scale.parameters())

#     if config.training.use_openclip_optimizer_scheduler:
#         optimizer = torch.optim.AdamW(
#             params,
#             lr=config.training.lr,
#             betas=(config.training.beta1, config.training.beta2),
#             eps=config.training.eps,
#         )
#         scheduler = cosine_lr(optimizer, config.training.lr, config.training.warmup, len(train_loader) * 16)
#     else:
#         optimizer = torch.optim.AdamW(params, lr=config.training.lr)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step, gamma=config.training.lr_decay_rate)

#     trainer = Trainer(config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader)
#     trainer.train()

#     if config.wandb_key is not None:
#         wandb.finish()
#     cleanup()

if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    # config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
    
    world_size = cli_args.ngpu
    mp.spawn(
        main,
        args=(world_size, cli_args, extras),
        nprocs=world_size
    )


# def main_old(rank, world_size, cli_args, extras):
#     print("rank: ", rank)
#     setup(rank, world_size)

#     config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
#     setup_cache(config)
#     print("config: ", config)

#     if config.autoresume:
#         config.trial_name = config.get('trial_name') + "@autoresume"
#     else:
#         config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
#     config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
#     config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

#     if rank == 0:
#         os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
#         os.makedirs(config.ckpt_dir, exist_ok=True)
#         if os.path.exists(config.code_dir):
#             shutil.rmtree(config.code_dir)
#         shutil.copytree("./src", config.code_dir)
    
#     config.device = 'cuda:{0}'.format(rank) 
    
#     if rank == 0:
#         config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
#         config.log_level = logging.DEBUG if config.debug else logging.INFO
#         setup_logging(config.log_path, config.log_level)
#         dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
#         logging.info("Using {} GPU(s).".format(config.ngpu))
#         if config.wandb_key is not None:
#             wandb.login(key=config.wandb_key)
#             wandb.init(project=config.project_name, name=config.trial_name, config=OmegaConf.to_object(config))

#     if config.train:
#         model = models.make(config).to(config.device)
#         if rank == 0:
#             total_params = sum(p.numel() for p in model.parameters())
#             logging.info(model)
#             logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))
        
#         torch.cuda.set_device(rank)
#         model.cuda(rank)
#         model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
#         # if config.model.name.startswith('Mink'):
#         #     model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
#         #     logging.info("Using MinkowskiSyncBatchNorm")
#         # else:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         logging.info("Using SyncBatchNorm")

#         logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
#         image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
#         text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)

#         logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)

#         train_loader = data.make(config, 'train', rank, world_size)
#         print("train_loader size: ", len(train_loader))
#         print("train_loader dataset size: ", len(train_loader.dataset))

#         if rank == 0:
#             modelnet40_loader = data.make_modelnet40test(config)
#             objaverse_lvis_loader = data.make_objaverse_lvis(config)
#             scanobjectnn_loader = data.make_scanobjectnntest(config)
#             # print("modelnet40_loader size: ", len(modelnet40_loader.dataset))
#             print("objaverse_lvis_loader size: ", len(objaverse_lvis_loader))
#             print("objaverse_lvis_loader dataset size: ", len(objaverse_lvis_loader.dataset))
#             # print("scanobjectnn_loader size: ", len(scanobjectnn_loader.dataset))
#         else:
#             modelnet40_loader = None
#             objaverse_lvis_loader = None
#             scanobjectnn_loader = None

#         if rank == 0:
#             if train_loader is not None:
#                 logging.info("Train iterations: {}".format(len(train_loader)))

#         if config.training.freeze_layers:
#             # Step 1: Freeze all parameters
#             for name, param in model.named_parameters():
#                 param.requires_grad = False
#                 # logging.info(f"Freezing {name}")
            
#             # Step 2: Unfreeze the 11th layer of the transformer
#             for name, param in model.named_parameters():
#                 # if name.find("ppat.transformer.layers.11") != -1:
#                 #     param.requires_grad = True
#                 #     logging.info(f"Unfreezing {name}")
#                 if name.find("proj") != -1:
#                     param.requires_grad = True
#                     logging.info(f"Unfreezing {name}")

#             # for name, param in model.ppat.transformer.layers[10].named_parameters():
#             #     param.requires_grad = True
#             #     logging.info(f"Unfreezing {name}")

#             # Step 3: Unfreeze the projection layer
#             # for name, param in model.proj.named_parameters():
#             #     param.requires_grad = True
#             #     logging.info(f"Unfreezing {name}")

#         params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + list(logit_scale.parameters()) 
#         if config.training.use_openclip_optimizer_scheduler:
#             optimizer = torch.optim.AdamW(
#                 params,
#                 lr=config.training.lr,
#                 betas=(config.training.beta1, config.training.beta2),
#                 eps=config.training.eps,
#             )
#             scheduler = cosine_lr(optimizer, config.training.lr, config.training.warmup, len(train_loader) * 16)
#         else:
#             optimizer = torch.optim.AdamW(params, lr=config.training.lr)
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step, gamma=config.training.lr_decay_rate)


#         trainer = Trainer(rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, \
#                         train_loader, modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader)


#         if config.resume is not None:
#             trainer.load_from_checkpoint(config.resume)
#         elif config.autoresume:
#             if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
#                 trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))
#         elif config.load_from_online:
#             model_name = "OpenShape/openshape-pointbert-vitg14-rgb"
#             trainer.load_from_online(model_name=model_name)

#         trainer.train()

#     if rank == 0 and config.wandb_key is not None:
#         wandb.finish()
#     cleanup()
