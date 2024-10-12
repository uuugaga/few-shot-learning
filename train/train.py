import torch
import torch.optim as optim
from utils.data_loader import get_dataloaders
from train.strategy import *
import train.strategy
from lion_pytorch import Lion
from tqdm import tqdm
import os
import numpy as np
import importlib


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['experiment']['num_classes']
    input_channels = config['model']['input_channels']
    epochs = config['training']['epochs']
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"

    # Load data
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize model
    model_module = importlib.import_module(f'models.{config["model"]["file"]}')
    model_class = getattr(model_module, config['model']['name'])
    model = model_class(num_classes=num_classes, input_channels=input_channels).to(device)

    # Initialize optimizer
    if config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    elif config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'] == 'Lion':
        optimizer = Lion(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    else:
        raise ValueError("Unknown optimizer type")
    
    # Initialize training strategy from config
    strategy_class = getattr(train.strategy, config['model']['strategy'])
    training_strategy = strategy_class()

    # Train the model
    best_val_loss = np.inf
    patience = config['training']['early_stopping']
    counter = 0

    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch: [{epoch+1}/{epochs}]", ncols=70, leave=False) as progress_bar:
            model.train()
            epoch_loss = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = training_strategy.train(model, data, labels, num_classes)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.update(1)

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                loss = training_strategy.train(model, data, labels, num_classes)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        tqdm.write(f"Epoch: [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(model.state_dict(), weight_path)
        else:
            counter += 1
            if counter >= patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                break
    
