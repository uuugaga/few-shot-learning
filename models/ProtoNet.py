# models/ProtoNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, num_classes=None, input_channels=None):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def compute_prototypes(embeddings, labels, num_classes):
        prototypes = []
        for c in range(num_classes):
            class_indices = (labels == c).nonzero(as_tuple=True)[0]
            class_embeddings = embeddings[class_indices]
            if class_embeddings.size(0) > 0:
                class_prototype = class_embeddings.mean(dim=0)
            else:
                class_prototype = torch.zeros(embeddings.size(1), device=embeddings.device)
            prototypes.append(class_prototype)
        return torch.stack(prototypes)