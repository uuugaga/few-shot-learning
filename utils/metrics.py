import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(predictions, labels):

    predictions, labels = np.array(predictions.to('cpu')), np.array(labels.to('cpu'))

    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, predictions)

    # Precision
    metrics['precision'] = precision_score(labels, predictions, average='weighted', zero_division=0)

    # Recall
    metrics['recall'] = recall_score(labels, predictions, average='weighted', zero_division=0)

    # F1 Score
    metrics['f1_score'] = f1_score(labels, predictions, average='weighted', zero_division=0)

    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
        
    return metrics

def get_class_accuary(confusion, class_names):
    class_accuracies = {}
    for idx, class_name in enumerate(class_names):
        # True Positives for the current class
        true_positive = confusion[idx, idx]
        # Total instances of the current class (True Positives + False Negatives)
        total = confusion[idx, :].sum()
        # Calculate accuracy for the current class
        class_accuracies[class_name] = true_positive / total if total > 0 else 0.0

    return class_accuracies



# Example usage
if __name__ == '__main__':
    # Randomly generated data for demonstration purposes
    labels = np.array([1, 0, 2, 1, 0])
    predictions = np.array([1, 0, 1, 1, 0])

    metrics_result = calculate_metrics(predictions, labels)
    
    print(metrics_result)
    print("Accuracy:", metrics_result['accuracy'])