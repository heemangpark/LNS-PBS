import torch
import torch.nn as nn
import dgl

AG_TYPE = 1
TASK_TYPE = 2

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

    def forward(self, g: dgl.DGLGraph, nf, target_node_idx):
        g.ndata['nf'] = nf
        g.pull()



