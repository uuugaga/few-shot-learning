import torch
import torch.optim as optim
from utils.data_loader import get_dataloaders
from train.strategy import *
import importlib
from lion_pytorch import Lion
from tqdm import tqdm
import os
import numpy as np


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(config, model):
    optimizers = {
        'Adam': lambda: optim.Adam(model.parameters(), lr=config['training']['learning_rate']),
        'AdamW': lambda: optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay']),
        'Lion': lambda: Lion(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    }
    optimizer_name = config['training']['optimizer']
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_name}")
    return optimizers[optimizer_name]()


def initialize_model(config, device):
    model_module = importlib.import_module(f'models.{config["model"]["file"]}')
    model_class = getattr(model_module, config['model']['name'])
    return model_class(config).to(device)


def save_best_model(model, weight_path, best_val_acc, val_acc):
    if val_acc > best_val_acc:
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        return val_acc, 0
    return best_val_acc, 1


def train_epoch(model, train_loader, optimizer, training_strategy, config, device):
    model.train()
    epoch_loss = 0.0
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    for i, (data, labels) in enumerate(tqdm(train_loader, desc="Training", ncols=65, leave=False)):
        data, labels = data.to(device), labels.to(device)
        loss = training_strategy.train(model, data, labels, config) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
    return epoch_loss / len(train_loader)


def validate_model(model, val_loader, training_strategy, config, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation", ncols=65, leave=False):
            labels = labels.to(device)
            preds = training_strategy.test(model, data, labels, config)
            preds = torch.tensor(preds) if not isinstance(preds, torch.Tensor) else preds
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(config):
    device = get_device()
    epochs, patience = config['training']['epochs'], config['training']['early_stopping']
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"

    train_loader, val_loader, _ = get_dataloaders(config)
    model = initialize_model(config, device)
    optimizer = get_optimizer(config, model)
    try:
        training_strategy = getattr(importlib.import_module('train.strategy'), config['model']['strategy'])()
    except AttributeError:
        raise ValueError(f"Training strategy '{config['model']['strategy']}' not found in 'train.strategy' module.")

    best_val_acc, counter = 0.0, 0
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, training_strategy, config, device)
        val_acc = validate_model(model, val_loader, training_strategy, config, device)

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val acc: {val_acc:.4f}")
        best_val_acc, counter = save_best_model(model, weight_path, best_val_acc, val_acc) if val_acc > best_val_acc else (best_val_acc, counter + 1)

        if counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break