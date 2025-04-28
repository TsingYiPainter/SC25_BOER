import torch

def GooLeNet():
    return  torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
