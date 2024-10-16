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

class PrototypicalBatchSampler(object):
    def __init__(self, config, labels, mode='train'):
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = config['experiment'][f'{mode}_n_ways']
        self.n_shots = config['experiment']['n_shots']
        self.num_query = config['experiment']['num_query']
        self.iterations = config['experiment']['iterations']
        self.mode = mode

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.indexes = [torch.tensor(np.where(self.labels == cls)[0]) for cls in self.classes]

        if len(self.classes) == self.classes_per_it and (self.mode == 'test' or self.mode == 'val'):
            self.fixed_support_idxs = []
            for idxs in self.indexes:
                permuted_idxs = idxs[torch.randperm(len(idxs))[:self.n_shots]]
                self.fixed_support_idxs.append(permuted_idxs)

            self.remaining_query_idxs = []
            for c, support_idxs in enumerate(self.fixed_support_idxs):
                idxs = self.indexes[c]
                remaining_idxs = torch.tensor([idx for idx in idxs if idx not in support_idxs])
                self.remaining_query_idxs.append(remaining_idxs)

            self.remaining_query_idxs = torch.cat(self.remaining_query_idxs)
            self.fixed_support_idxs = torch.cat(self.fixed_support_idxs)

    def __iter__(self):
        if len(self.classes) == self.classes_per_it and (self.mode == 'test' or self.mode == 'val'):
            num_iterations = len(self.remaining_query_idxs) // (self.n_shots * self.classes_per_it)
            for i in range(num_iterations):

                query_len = self.n_shots * self.classes_per_it
                support_batch = self.fixed_support_idxs
                query_batch = self.remaining_query_idxs[i * query_len: (i + 1) * query_len]

                if len(support_batch) < len(query_batch):
                    support_batch = torch.cat([support_batch, support_batch.repeat((len(query_batch) - len(support_batch)) // len(support_batch))])

                batch = []
                for i in range(len(support_batch)):
                    batch.append((support_batch[i], query_batch[i]))

                yield batch

        else:
            for _ in range(self.iterations):
                chosen_classes = torch.multinomial(torch.ones(len(self.classes)), self.classes_per_it, replacement=False)
                support_batch, query_batch = [], []

                for c in chosen_classes:
                    idxs = self.indexes[c]
                    permuted_idxs = idxs[torch.randperm(len(idxs))[:self.n_shots + self.num_query]]
                    support_batch.append(permuted_idxs[:self.n_shots])
                    query_batch.append(permuted_idxs[self.n_shots:])

                support_batch = torch.cat(support_batch)
                query_batch = torch.cat(query_batch)

                if len(support_batch) < len(query_batch):
                    support_batch = torch.cat([support_batch, support_batch.repeat((len(query_batch) - len(support_batch)) // len(support_batch))])

                batch = []
                for i in range(len(support_batch)):
                    batch.append((support_batch[i], query_batch[i]))
                yield batch

    def __len__(self):
        if len(self.classes) == self.classes_per_it and (self.mode == 'test' or self.mode == 'val'):
            return len(self.remaining_query_idxs) // (self.n_shots * self.classes_per_it)
        return self.iterations  