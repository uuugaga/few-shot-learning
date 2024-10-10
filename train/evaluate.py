import torch
from models.ProtoNet import ProtoNet
from utils.data_loader import get_dataloaders

def evaluate_model(model_name, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 50
    batch_size = 32
    weight_path = './weight/protonet.pth'

    # Load data
    _, test_loader = get_dataloaders(batch_size, num_classes)

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
    print(f"Model {model_name} Accuracy: {accuracy:.4f}")
    return accuracy