import torch

def load_from_local_checkpoint(model, path="pretrained/openshape-pointbert-vitg14-rgb/checkpoint.pt"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model
