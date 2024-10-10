import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random

# Define the Prototypical Network model
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
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

# Compute the prototype representations of each class
def compute_prototypes(embeddings, labels, num_classes):
    prototypes = []
    for c in range(num_classes):
        class_indices = (labels == c).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_indices]
        class_prototype = class_embeddings.mean(dim=0)
        prototypes.append(class_prototype)
    return torch.stack(prototypes)

# Train the Prototypical Network
def ensure_all_classes_present(batch, num_classes):
    data, labels = zip(*batch)
    data = torch.stack(data)
    labels = torch.tensor(labels)
    unique_labels = labels.unique()
    missing_classes = [c for c in range(num_classes) if c not in unique_labels]
    if missing_classes:
        for c in missing_classes:
            data = torch.cat([data, data[0:1]])  # Duplicate the first sample
            labels = torch.cat([labels, torch.tensor([c])])
    return data, labels

# Train the Prototypical Network
def train_protonet(model, optimizer, train_loader, num_classes, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(data)
            prototypes = compute_prototypes(embeddings, labels, num_classes)

            # Compute distances between query points and prototypes
            dists = torch.cdist(embeddings, prototypes, p=2)

            # Compute negative log-probability loss
            log_prob = F.log_softmax(-dists, dim=1)
            loss = F.nll_loss(log_prob, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluate the Prototypical Network
def evaluate_protonet(model, test_loader, num_classes):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)
            prototypes = compute_prototypes(embeddings, labels, num_classes)
            dists = torch.cdist(embeddings, prototypes, p=2)
            preds = torch.argmin(dists, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

# Example usage
if __name__ == "__main__":
    # Use CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    classes = list(range(5))
    class_counts = {c: 0 for c in classes}
    max_samples_per_class = 50
    train_indices = []
    for i, label in enumerate(train_dataset.targets):
        if label in classes and class_counts[label] < max_samples_per_class:
            train_indices.append(i)
            class_counts[label] += 1
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=lambda batch: ensure_all_classes_present(batch, num_classes))
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_classes = len(classes)

    train_protonet(model, optimizer, train_loader, num_classes, epochs=4)

    # Evaluate the model
    evaluate_protonet(model, test_loader, num_classes)