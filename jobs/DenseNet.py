import torch
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
# # or any of these variants
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
# # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
# model.eval()

def DenseNet():
    return  torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)