import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# I realised that the test is a bit unfair for annotation budget <50%
# I'll add an option for loading a pretrained imagenet model and adapting
# it to CIFAR100


def load_resnet18(
    num_classes: int = 100, with_pretrained_weights: bool = True
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if with_pretrained_weights else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
