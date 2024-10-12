# utils/data_loader.py
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

def get_dataloaders(config):
    
    batch_size = config['training']['batch_size']
    num_classes = config['experiment']['n_ways']
    dataset_name = config['experiment']['dataset']
    train_split = config['data']['split']['train']
    val_split = config['data']['split']['val']
    test_split = config['data']['split']['test']

    assert (train_split + val_split + test_split) == 1.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        train_dataset = datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
    else:
        raise ValueError(f'Unknow dataset:{dataset_name}')


    class_counts = {c: 0 for c in range(num_classes)}
    max_samples_per_class = config['experiment']['n_shots']
    train_indices = []
    for i, label in enumerate(train_dataset.targets):
        if label in class_counts and class_counts[label] < max_samples_per_class:
            train_indices.append(i)
            class_counts[label] += 1

    train_subset = Subset(train_dataset, train_indices)
    
    total_train_samples = len(train_subset)
    val_size = int(total_train_samples * val_split)
    train_size = total_train_samples - val_size
    train_subset, val_subset = random_split(train_subset, [train_size, val_size])

    test_subset = Subset(test_dataset, [i for i, label in enumerate(test_dataset.targets) if label < num_classes])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
