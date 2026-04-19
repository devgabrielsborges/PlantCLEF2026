import timm
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL) implementation.
    The forward pass is identity, while the backward pass negates the gradient
    and multiplies it by a scaling factor alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # The backward pass returns the negated gradient for the first input (x)
        # and None for the second input (alpha) since alpha doesn't require gradients.
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    """
    Helper function to apply the Gradient Reversal Layer.
    """
    return GradientReversal.apply(x, alpha)


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) implementation.
    Consists of a shared feature extractor, a label predictor, and a domain discriminator.
    """

    def __init__(self, backbone_name, num_classes):
        super(DANN, self).__init__()
        # 1. Feature Extractor
        self.feature_extractor = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )

        # Get feature dimension dynamically
        self.num_features = self.feature_extractor.num_features
        feature_dim = self.num_features

        # 2. Label Predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # 3. Domain Discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, alpha=1.0):
        # Extract features (e.g., [CLS] token from ViT)
        features = self.feature_extractor(x)

        # Branch 1: Label Prediction
        class_output = self.label_predictor(features)

        # Branch 2: Domain Discrimination with GRL
        reverse_features = grad_reverse(features, alpha)
        domain_output = self.domain_discriminator(reverse_features)

        return class_output, domain_output
