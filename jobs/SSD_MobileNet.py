import torch


def SSD_MobileNet():
    return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')