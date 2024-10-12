import torch
import torch.nn.functional as F
from models.ProtoNet import ProtoNet

class Strategy:
    def train(self, model, data, labels):
        raise NotImplementedError

class Prototypical(Strategy):
    def train(self, model, data, labels, num_classes):
        # Prototypical Network-specific training steps
        embeddings = model(data)
        embeddings = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)
        prototypes = model.compute_prototypes(embeddings, labels, num_classes)
        dists = torch.cdist(embeddings, prototypes, p=2)
        log_prob = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_prob, labels)
        return loss

    def test(self, model, data, labels, num_classes):
        # Prototypical Network-specific testing steps
        embeddings = model(data)
        embeddings = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)
        prototypes = model.compute_prototypes(embeddings, labels, num_classes)
        dists = torch.cdist(embeddings, prototypes, p=2)
        preds = torch.argmin(dists, dim=1)
        return preds

class Standard(Strategy):
    def train(self, model, data, labels, num_classes):
        # Standard training steps (e.g., classification)
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def test(self, model, data, labels, num_classes):
        # Standard testing steps (e.g., classification)
        outputs = model(data)
        preds = torch.argmax(outputs, dim=1)
        return preds