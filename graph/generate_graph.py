from copy import deepcopy

import networkx as nx
import numpy as np


def graph(world, rand_coord=False):
    g = gen_graph(world, rand_coord=rand_coord)
    components = [c for c in nx.connected_components(g)]
    if len(components) != 1:
        while len(components) == 1:
            g = gen_graph(world, rand_coord=rand_coord)
            components = [c for c in nx.connected_components(g)]

    return g


def gen_graph(world, rand_coord=False):
    world = deepcopy(world)
    m, n = world.shape[0], world.shape[1]
    g = nx.grid_2d_graph(m, n)

    if rand_coord:
        rand_interval_x = np.random.random(m - 1) + .5
        rand_x = np.array([rand_interval_x[:i].sum() for i in range(m)])
        rand_x /= rand_x.sum()

        rand_interval_y = np.random.random(n - 1) + .5
        rand_y = np.array([rand_interval_y[:i].sum() for i in range(n)])
        rand_y /= rand_y.sum()

    else:
        rand_x = np.array([0 + i / (n - 1) for i in range(m)])
        rand_y = np.array([0 + i / (m - 1) for i in range(n)])

    xs = np.array(list(rand_x) * n)
    ys = rand_y[::-1].repeat(m)

    for id, n_id in enumerate(g.nodes()):
        g.nodes[n_id]['loc'] = np.stack([xs, ys], -1)[id].tolist()
        g.nodes[n_id]['type'] = int(world.reshape(-1)[id])

    for r, c in zip(world.nonzero()[0], world.nonzero()[1]):
        g.remove_node((r, c))

    for id, e_id in enumerate(g.edges()):
        loc = (np.array(g.nodes[e_id[0]]['loc']) - np.array(g.nodes[e_id[1]]['loc'])) ** 2
        dist = loc.sum(-1).reshape(-1, 1) ** .5
        g.edges[e_id]['dist'] = dist

    return g
