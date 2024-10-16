import torch
import torch.nn.functional as F
from models.ProtoNet import ProtoNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Strategy:
    def train(self, model, data, labels, config):
        raise NotImplementedError("train method must be implemented in subclass")

    def test(self, model, data, labels, config):
        raise NotImplementedError("test method must be implemented in subclass")


class Prototypical(Strategy):
    def train(self, model, data, labels, config):
        support_data, query_data, support_labels = [d.to(device) for d in data]
        
        # Remap labels to start from 0 to n_classes
        support_labels, labels = self._remap_labels(support_labels, labels)

        # Extract embeddings
        support_embeddings, query_embeddings = model(support_data), model(query_data)

        # Compute prototypes
        support_prototypes = self._compute_prototypes(support_embeddings, support_labels)

        # Calculate distances and loss
        log_p_y = self._calculate_log_probabilities(query_embeddings, support_prototypes)
        loss = F.nll_loss(log_p_y, labels)

        return loss

    def test(self, model, data, labels, config):
        support_data, query_data, support_labels = [d.to(device) for d in data]

        # Remap labels to start from 0 to n_classes
        support_labels, labels = self._remap_labels(support_labels, labels)

        with torch.no_grad():
            # Extract embeddings
            support_embeddings, query_embeddings = model(support_data), model(query_data)

            # Compute prototypes and distances
            support_prototypes = self._compute_prototypes(support_embeddings, support_labels)
            preds = self._get_predictions(query_embeddings, support_prototypes)

        return preds, labels

    @staticmethod
    def _remap_labels(support_labels, labels):
        classes, _ = torch.sort(torch.unique(support_labels))
        label_map = {int(c): i for i, c in enumerate(classes)}
        remapped_support_labels = torch.tensor([label_map[c.item()] for c in support_labels], device=device)
        remapped_labels = torch.tensor([label_map[c.item()] for c in labels], device=device)
        return remapped_support_labels, remapped_labels

    @staticmethod
    def _compute_prototypes(embeddings, labels):
        classes = torch.unique(labels)
        return torch.vstack([torch.mean(embeddings[labels == c], dim=0) for c in classes])

    @staticmethod
    def _calculate_log_probabilities(query_embeddings, support_prototypes):
        dists = torch.cdist(query_embeddings, support_prototypes, p=2)
        return F.log_softmax(-dists, dim=1)

    @staticmethod
    def _get_predictions(query_embeddings, support_prototypes):
        dists = torch.cdist(query_embeddings, support_prototypes, p=2)
        return torch.argmin(dists, dim=1)


class Standard(Strategy):
    def train(self, model, data, labels, config):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def test(self, model, data, labels, config):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        preds = torch.argmax(outputs, dim=1)
        return preds, labels


# Factory class to easily select strategy
class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_type):
        strategies = {
            "Prototypical": Prototypical,
            "Standard": Standard
        }
        if strategy_type in strategies:
            return strategies[strategy_type]()
        raise ValueError(f"Unknown strategy type: {strategy_type}")