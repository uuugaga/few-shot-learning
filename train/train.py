import torch
import torch.optim as optim
from models.ProtoNet import ProtoNet
from utils.data_loader import get_dataloaders
from train.TrainingStrategy import PrototypicalTrainingStrategy
import os

def train_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 50
    batch_size = 32
    epochs = 20
    weight_path = './weight/protonet.pth'

    # Load data
    train_loader, _ = get_dataloaders(batch_size, num_classes)

    # Initialize model
    model = ProtoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_strategy = PrototypicalTrainingStrategy()

    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            # Train using the strategy
            loss = training_strategy.train(model, optimizer, data, labels, num_classes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(model.state_dict(), weight_path)
    return weight_path