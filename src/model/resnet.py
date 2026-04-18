import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# I realised that the test is a bit unfair for annotation budget <50%
# I'll add an option for loading a pretrained imagenet model and adapting
# it to CIFAR100


def load_resnet18(
    num_classes: int = 100,
    with_pretrained_weights: bool = True,
    strip_fc: bool = False,
) -> nn.Module:
    """
    Load a ResNet-18 model configured for classification or feature extraction.

    Replaces the final fully connected layer with one matching num_classes.
    If strip_fc is True, replaces the FC layer with nn.Identity instead,
    returning raw 512-dimensional embeddings for use as a feature extractor.

    Arguments
    ---------
    num_classes : int
        Number of output classes for the classification head. Default: 100.
    with_pretrained_weights : bool
        If True, loads ImageNet-pretrained weights. Default: True.
    strip_fc : bool
        If True, removes the FC layer for use as a backbone extractor.
        Default: False.

    Returns
    -------
    nn.Module
        Configured ResNet-18 model.

    Example
    -------
    >>> model = load_resnet18(num_classes=100, with_pretrained_weights=True)
    >>> extractor = load_resnet18(with_pretrained_weights=True, strip_fc=True)
    """
    weights = ResNet18_Weights.DEFAULT if with_pretrained_weights else None
    model = models.resnet18(weights=weights)
    model.fc = (
        nn.Linear(model.fc.in_features, num_classes)
        if not strip_fc
        else nn.Identity()
    )
    return model
