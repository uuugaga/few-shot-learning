import torch
import torch.nn.functional as F
from models.ProtoNet import ProtoNet

class TrainingStrategy:
    def train(self, model, data, labels):
        raise NotImplementedError

class PrototypicalTrainingStrategy(TrainingStrategy):
    def train(self, model, data, labels, num_classes):
        # Prototypical Network-specific training steps
        embeddings = model(data)
        embeddings = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)
        prototypes = model.compute_prototypes(embeddings, labels, num_classes)
        dists = torch.cdist(embeddings, prototypes, p=2)
        log_prob = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_prob, labels)
        return loss

class StandardTrainingStrategy(TrainingStrategy):
    def train(self, model, data, labels):
        # Standard training steps (e.g., classification)
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        return loss