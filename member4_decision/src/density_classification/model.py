import torch.nn as nn
from torchvision import models


class DensityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(DensityClassifier, self).__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)