import torch
import torch.nn as nn
from torchvision import models
# https://pytorch.org/vision/stable/models.html
from torchsummary import summary


# edit model to fit smartphone images
class PokemonResNet(nn.Module):
    def __init__(self, num_classes=879, pretrained=True, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet34
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        # Replace the final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size=(3, 224, 224)):
        summary(self.model, input_size)


if __name__ == "__main__":
    summary(PokemonResNet(), input_size=(3, 224, 224))
