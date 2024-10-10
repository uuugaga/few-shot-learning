# utils/data_loader.py
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

def get_dataloaders(batch_size, num_classes):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    class_counts = {c: 0 for c in range(num_classes)}
    max_samples_per_class = {c: random.choice([5, 20]) for c in range(num_classes)}
    train_indices = []
    for i, label in enumerate(train_dataset.targets):
        if label in class_counts and class_counts[label] < max_samples_per_class[label]:
            train_indices.append(i)
            class_counts[label] += 1

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, [i for i, label in enumerate(test_dataset.targets) if label < num_classes])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader