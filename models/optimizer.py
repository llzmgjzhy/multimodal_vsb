import torch


def get_optimizer(name):

    if name == "Adam":
        return torch.optim.Adam
    if name == "AdamW":
        return torch.optim.AdamW
    elif name == "RAdam":
        return torch.optim.RAdam
