import torch
import re
from collections import OrderedDict
from huggingface_hub import hf_hub_download
# from minlora import add_lora

import models

def load_model(config, model_name="OpenShape/openshape-pointbert-vitg14-rgb"):
    print("Loading model: ", model_name)
    model = models.make(config).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    # model.to(device)
    return model

def load_model_no_classification(config, model_name="OpenShape/openshape-pointbert-vitg14-rgb"):
    print("Loading model: ", model_name)
    model = models.make_no_classification(config).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k) and not re.search("proj", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    # model.to(device)
    return model

def load_model_from_path(config, model_path, device):
    print("Loading model from path: ", model_path)
    model = models.make(config)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint = torch.load(model_path)
    print("epoch: ", checkpoint['epoch'])

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v

    model.load_state_dict(model_dict)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model

# def load_lora_model_from_path(config, model_path, device):
#     print("Loading model from path: ", model_path)
#     model = models.make(config)
#     add_lora(model)
#     # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     checkpoint = torch.load(model_path)

#     # model_dict = OrderedDict()
#     # pattern = re.compile('module.')
#     # for k,v in checkpoint['state_dict'].items():
#     #     if re.search("module", k):
#     #         model_dict[re.sub(pattern, '', k)] = v

#     # model.load_state_dict(model_dict)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)

#     return model

def calc_accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))
            # print("correct: ", correct.shape)
            # print("correct: ", correct)
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct