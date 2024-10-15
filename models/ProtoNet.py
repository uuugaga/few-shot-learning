import torch
import torch.nn as nn
import numpy as np


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    def __init__(self, config):
        super(ProtoNet, self).__init__()
        self.input_channels = config['model']['input_channels']
        self.num_layers = config['model'].get('num_layers', 4)
        self.num_filters = config['model'].get('num_filters', [64] * self.num_layers)

        layers = []
        in_channels = self.input_channels
        for out_channels in self.num_filters:
            layers.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    @staticmethod
    def compute_prototypes(embeddings, labels, num_classes):
        prototypes = [
            embeddings[(labels == c).nonzero(as_tuple=True)[0]].mean(dim=0) if (labels == c).any()
            else torch.zeros(embeddings.size(1), device=embeddings.device)
            for c in range(num_classes)
        ]
        return torch.stack(prototypes)


class PrototypicalBatchSampler(object):
    def __init__(self, config, labels, mode='train'):
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = config['experiment'][f'{mode}_n_ways']
        self.sample_per_class = config['experiment']['n_shots'] + config['experiment']['num_query']
        self.iterations = config['experiment']['iterations']

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.indexes = [torch.tensor(np.where(self.labels == cls)[0]) for cls in self.classes]

    def __iter__(self):
        for _ in range(self.iterations):
            chosen_classes = torch.multinomial(torch.ones(len(self.classes)), self.classes_per_it, replacement=False)
            batch = torch.cat([
                self.indexes[c][torch.randperm(len(self.indexes[c]))[:self.sample_per_class]]
                for c in chosen_classes
            ])
            yield batch

    def __len__(self):
        return self.iterations


class PrototypicalBatchSamplerSupportQuerySplit(object):
    def __init__(self, config, labels, mode='train'):
        super(PrototypicalBatchSamplerSupportQuerySplit, self).__init__()
        self.labels = labels
        self.classes_per_it = config['experiment'][f'{mode}_n_ways']
        self.n_shots = config['experiment']['n_shots']
        self.num_query = config['experiment']['num_query']
        self.iterations = config['experiment']['iterations']

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.indexes = [torch.tensor(np.where(self.labels == cls)[0]) for cls in self.classes]

    def __iter__(self):
        for _ in range(self.iterations):
            chosen_classes = torch.multinomial(torch.ones(len(self.classes)), self.classes_per_it, replacement=False)
            support_batch, query_batch, support_labels, query_labels = [], [], [], []

            for c in chosen_classes:
                idxs = self.indexes[c]
                permuted_idxs = idxs[torch.randperm(len(idxs))[:self.n_shots + self.num_query]]
                support_batch.append(permuted_idxs[:self.n_shots])
                query_batch.append(permuted_idxs[self.n_shots:])
                support_labels.extend([self.classes[c]] * self.n_shots)
                query_labels.extend([self.classes[c]] * self.num_query)

            support_batch = torch.cat(support_batch)
            query_batch = torch.cat(query_batch)
            batch = []
            for i in range(len(support_batch)):
                batch.append((support_batch[i], query_batch[i]))
            yield batch

    def __len__(self):
        return self.iterations