import torch
import torch.nn as nn
from torchvision import models

class RoomClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Backbone not implemented")
        
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)