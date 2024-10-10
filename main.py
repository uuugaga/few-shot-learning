# main.py
import argparse
from train.train import train_model
from train.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--model_name", type=str, help="The name of the model to train and evaluate")
    args = parser.parse_args()

    # Train the model
    weight_path = train_model(args.model_name)

    # Evaluate the model
    evaluate_model(args.model_name, weight_path)