import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
