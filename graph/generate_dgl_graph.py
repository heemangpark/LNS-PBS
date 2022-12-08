from copy import deepcopy

import dgl
import numpy as np
import torch


def workspace_graph(world, rand_coord=True, four_dir=True):
    world = deepcopy(world)
    m, n = world.shape
    g = dgl.graph(list())
    g.add_nodes(m * n)

    if rand_coord:
        # node position
        rand_interval_x = torch.rand(m - 1) + .5
        rand_x = [rand_interval_x[:i].sum() for i in range(m)]
        rand_x = torch.Tensor(rand_x)
        rand_x /= rand_x.sum()

        rand_interval_y = torch.rand(n - 1) + .5
        rand_y = [rand_interval_y[:i].sum() for i in range(n)]
        rand_y = torch.Tensor(rand_y)
        rand_y /= rand_y.sum()
    else:
        rand_x = torch.Tensor([0 + i / (n - 1) for i in range(m)])
        rand_y = torch.Tensor([0 + i / (m - 1) for i in range(n)])

    xs = rand_x.repeat(n, 1).reshape(-1)
    ys = rand_y.flip(-1).repeat(m, 1).T.reshape(-1)

    g.ndata['loc'] = torch.stack([xs, ys], -1)
    g.ndata['type'] = torch.Tensor(world.reshape(-1, 1))

    # add edge
    matrix = np.arange(m * n).reshape(m, -1)
    v_from = matrix[:-1].reshape(-1)
    v_to = matrix[1:].reshape(-1)

    g.add_edges(v_from, v_to)
    g.add_edges(v_to, v_from)

    h_from = matrix[:, :-1].reshape(-1)
    h_to = matrix[:, 1:].reshape(-1)

    g.add_edges(h_from, h_to)
    g.add_edges(h_to, h_from)

    if not four_dir:
        dig_from = matrix[:-1, :-1].reshape(-1)
        dig_to = matrix[1:, 1:].reshape(-1)
        g.add_edges(dig_from, dig_to)
        g.add_edges(dig_to, dig_from)

        ddig_from = matrix[1:, :-1].reshape(-1)
        ddig_to = matrix[:-1, 1:].reshape(-1)
        g.add_edges(ddig_from, ddig_to)
        g.add_edges(ddig_to, ddig_from)

    # compute ef
    g.apply_edges(lambda edges: {'dist': ((edges.src['loc'] - edges.dst['loc']) ** 2).sum(-1).reshape(-1, 1) ** .5})

    # remove obstacle
    obs_idx = world.reshape(-1).nonzero()[0]
    g.remove_nodes(obs_idx)

    return g
