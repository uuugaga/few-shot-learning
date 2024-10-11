import torch
from models.ProtoNet import ProtoNet
from utils.data_loader import get_dataloaders

def evaluate_model(config, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['experiment']['num_classes']
    batch_size = config['testing']['batch_size']
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"

    # Load data
    _, _, test_loader = get_dataloaders(config)

    # Initialize model and load weights
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)
            prototypes = model.compute_prototypes(embeddings, labels, num_classes)
            dists = torch.cdist(embeddings, prototypes, p=2)
            preds = torch.argmin(dists, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Model {config['model']['name']} Accuracy: {accuracy:.4f}")
    return accuracy
