import torch
import torch.nn as nn
import torch.nn.functional as F
import math

cfg = [64]

def make_layer(config):
    layers = []
    in_planes = 1
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_planes, value, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_planes = value
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, config=cfg, num_classes=1000, cifar=False):
        super(VGG, self).__init__()
        self.features = make_layer(config)
        
    def forward(self, x):
        out = self.features(x)
        return out
    
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# def VGG11(cifar=False):
#     return VGG(config = cfg['A'], cifar = cifar)

# def VGG13(cifar=False):
#     return VGG(config = cfg['B'], cifar = cifar)

# def VGG16(cifar=False):
#     return VGG(config = cfg['D'], cifar = cifar)

# def VGG19(cifar=False):
#     return VGG(config = cfg['E'], cifar = cifar)