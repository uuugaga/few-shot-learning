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
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)
        
        n_support = config['experiment']['n_shots']
        n_query = config['experiment']['num_query']

        support_idxs, query_idxs = self._get_support_query_indices(labels, n_support)
        support_embeddings = torch.stack([embeddings[idx].mean(0) for idx in support_idxs])

        # Batch the concatenation of query indices to improve performance
        batched_query_idxs = torch.cat(query_idxs, dim=0)
        query_embeddings = embeddings[batched_query_idxs]

        dists = torch.cdist(query_embeddings, support_embeddings, p=2)
        log_p_y = F.log_softmax(-dists, dim=1)

        target_inds = torch.arange(len(support_idxs), device=data.device).repeat_interleave(n_query)
        loss = F.nll_loss(log_p_y, target_inds)
        
        return loss

    def test(self, model, data, labels, config):
        support_data, query_data, support_labels = [d.to(device) for d in data]
        with torch.no_grad():
            support_embeddings, query_embeddings = model(support_data), model(query_data)
        
        support_prototypes = self._compute_prototypes(support_embeddings, support_labels)
        dists = torch.cdist(query_embeddings, support_prototypes, p=2)
        preds = torch.argmin(dists, dim=1)
        
        return preds
    
    @staticmethod
    def _get_support_query_indices(labels, n_support):
        classes = torch.unique(labels)
        support_idxs = [torch.where(labels == c)[0][:n_support] for c in classes]
        query_idxs = [torch.where(labels == c)[0][n_support:] for c in classes]
        return support_idxs, query_idxs
    
    @staticmethod
    def _compute_prototypes(embeddings, labels):
        classes = torch.unique(labels)
        return torch.vstack([torch.mean(embeddings[labels == c], dim=0) for c in classes])


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
        
        return preds


# Factory class to easily select strategy
class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_type):
        if strategy_type == "Prototypical":
            return Prototypical()
        elif strategy_type == "Standard":
            return Standard()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")