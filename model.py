import torchvision
import torch.nn as nn
import torch

class VGG16(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.model = torchvision.models.vgg16(pretrained=True)
        num_features = self.model.classifier[6].in_features
        classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            )
        self.model.classifier[6] = classifier
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.model(x)
        return out
    

class ResNet50(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.model(x)
        return out