import dgl
import torch
from torch import nn as nn

from nn.gnn import GNN


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.log_softmax = nn.LogSoftmax()

    def forward(self, input):
        batch = True if len(input.shape) == 3 else False
        x = self.layers(input)
        if batch:
            log_probs = self.log_softmax(x)
        else:
            log_probs = self.log_softmax(x.unsqueeze(0))

        return log_probs.squeeze()


class NLNS(nn.Module):
    def __init__(self, embedding_dim=64, gnn_layers=3, train=True):
        super(NLNS, self).__init__()
        self.device = 'cuda' if train else 'cpu'
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(2, embedding_dim)
        self.gnn = GNN(
            in_dim=embedding_dim,
            out_dim=embedding_dim,
            embedding_dim=embedding_dim,
            n_layers=gnn_layers,
            residual=True,
        )

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def train(self, task_idx=None, graphs=None, labels=None):
        batch_size = 10
        num_batches = len(task_idx) // batch_size

        for b in range(num_batches):
            t = task_idx[b * batch_size: (b + 1) * batch_size]
            g = dgl.batch(graphs[b * batch_size: (b + 1) * batch_size]).to(self.device)
            l = labels[b * batch_size: (b + 1) * batch_size]

            n_ag = sum(g.ndata['type'] == 1).item() // batch_size
            n_task = sum(g.ndata['type'] == 2).item() // batch_size

            nf = g.ndata['coord'].to(self.embedding.weight.dtype)
            n_embed = self.embedding(nf)
            out_nf = self.gnn(g, n_embed)
            out_nf = out_nf.reshape(batch_size, n_ag + n_task)

            # log_probs = self.policy(out_nf[n_ag:])
            log_probs = self.log_softmax(out_nf[:, n_ag:])
            label_probs = torch.Tensor(l).to(self.device)

            loss = self.loss(log_probs, label_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def eval(self, g=None, random_action=False, sample=True):
        nf = g.ndata['coord'].to(self.embedding.weight.dtype)
        n_embed = self.embedding(nf)
        out_nf = self.gnn(g, n_embed)

        n_ag = sum(g.ndata['type'] == 1).item()
        n_task = sum(g.ndata['type'] == 2).item()

        if random_action:
            rand_probs = torch.rand(n_task)
            log_probs = (rand_probs / sum(rand_probs)).log()
        else:
            log_probs = self.log_softmax(out_nf[n_ag:].unsqueeze(0)).squeeze()

        if sample:
            actions = torch.distributions.Categorical(logits=log_probs).sample((3,)).tolist()
        else:
            actions = torch.topk(log_probs, 3).indices.tolist()

        return list(set(actions))
