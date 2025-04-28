import timm

def MnasNet():
    return timm.create_model('mnasnet_100', pretrained=True)