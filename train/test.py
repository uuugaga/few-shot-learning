import torch
from utils.data_loader import get_dataloaders
import importlib
from tqdm import tqdm
import argparse
import yaml
from utils.metrics import calculate_metrics, get_class_accuary

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
    with torch.no_grad():
        preds_list, labels_list = [], []
        for data, labels in tqdm(test_loader, desc="Testing Model", ncols=65, leave=False):
            labels = labels.to(DEVICE)
            preds, labels = strategy.test(model, data, labels, config)
            preds_list.append(preds)
            labels_list.append(labels)
        
        metrics = calculate_metrics(torch.cat(preds_list), torch.cat(labels_list))
            
    return metrics

def test_model(config):
    """Test the model with the given configuration."""
    # Load data
    _, _, test_loader, classes_name = get_dataloaders(config)

    # Load model
    model = load_model(config)

    # Load training strategy
    strategy = load_strategy(config)

    # Evaluate model
    metrics = evaluate_model(model, test_loader, strategy, config)

    print(f"Model {config['model']['name']} Accuracy: {metrics['accuracy']:.4f}")

    # class_accuracy = get_class_accuary(metrics['confusion_matrix'], classes_name)
    # print(class_accuracy)

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