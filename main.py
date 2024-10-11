import argparse
import yaml
from train.train import train_model
from train.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--model_name", type=str, help="The name of the model to train and evaluate")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--num_classes", type=int, help="Number of classes in the dataset")
    args = parser.parse_args()

    # XOR check: either use config file or individual parameters, not both
    if args.config and (args.batch_size or args.epochs or args.learning_rate or args.num_classes):
        raise ValueError("Please specify either a config file or individual parameters, not both.")

    # Load configuration from file or default config
    if args.config:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        # Use individual command-line parameters
        config = {
            'model_name': args.model_name,
            'training': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate
            },
            'experiment': {
                'num_classes': args.num_classes
            }
        }

    weight_path = train_model(config)

    evaluate_model(config, weight_path)