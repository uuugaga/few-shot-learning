import torch
from utils.data_loader import get_dataloaders
import importlib
from tqdm import tqdm
import argparse
import yaml

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config):
    """Initialize and load the model from weight path."""
    model_module = importlib.import_module(f'models.{config["model"]["file"]}')
    model_class = getattr(model_module, config['model']['name'])
    model = model_class(config).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    weight_path = f"{config['paths']['weight_dir']}/{config['model']['name']}.pth"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
    except FileNotFoundError:
        raise FileNotFoundError(f"Weight file not found at {weight_path}. Please check the path and try again.")
    except RuntimeError as e:
        raise RuntimeError(f"Error loading the model weights: {str(e)}")
    model.eval()
    return model

def load_strategy(config):
    """Load the training strategy from config."""
    strategy_module = importlib.import_module(f'train.strategy')
    strategy_class = getattr(strategy_module, config['model']['strategy'])
    return strategy_class()

def evaluate_model(model, test_loader, strategy, config):
    """Evaluate the model on the test dataset."""
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Testing Model", ncols=65, leave=True) as progress_bar:
            for data, labels in test_loader:
                labels = labels.to(DEVICE)
                preds = strategy.test(model, data, labels, config)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                progress_bar.update(1)
    return correct / total

def test_model(config):
    """Test the model with the given configuration."""
    # Load data
    _, _, test_loader = get_dataloaders(config)

    # Load model
    model = load_model(config)

    # Load training strategy
    strategy = load_strategy(config)

    # Evaluate model
    test_acc = evaluate_model(model, test_loader, strategy, config)

    print(f"Model {config['model']['name']} Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test a trained model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Test the model
    test_model(config)