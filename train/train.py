import torch
import torch.optim as optim
from models.ProtoNet import ProtoNet
from utils.data_loader import get_dataloaders
from train.TrainingStrategy import *
from lion_pytorch import Lion
from tqdm import tqdm
import os


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['experiment']['num_classes']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"

    # Load data
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize model
    model = ProtoNet().to(device)

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
    training_strategy = globals()[config['strategy']['training_strategy']]()

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(epochs), desc=f"Epoch 1/{epochs}", ncols=70)

    # Train the model
    for epoch in progress_bar:
        model.train()
        epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = training_strategy.train(model, data, labels, num_classes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                loss = training_strategy.train(model, data, labels, num_classes)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)


        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        progress_bar.set_postfix(los=avg_loss)
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    return weight_path
