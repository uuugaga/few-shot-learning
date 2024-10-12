import torch
from utils.data_loader import get_dataloaders
import importlib
import train.strategy

def test_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['experiment']['num_classes']
    input_channels = config['model']['input_channels']
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"

    # Load data
    _, _, test_loader = get_dataloaders(config)

    # Initialize model
    model_module = importlib.import_module(f'models.{config["model"]["file"]}')
    model_class = getattr(model_module, config['model']['name'])
    model = model_class(num_classes=num_classes, input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    # Initialize training strategy from config
    strategy_class = getattr(train.strategy, config['model']['strategy'])
    training_strategy = strategy_class()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            preds = training_strategy.test(model, data, labels, num_classes)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Model {config['model']['name']} Accuracy: {accuracy:.4f}")
    return accuracy
