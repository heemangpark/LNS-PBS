from collections import Counter
from itertools import combinations

import numpy as np


def collision_detect(paths):
    max_len = max([len(p) for p in paths])
    n_ag = len(paths)
    collide = False
    collision_agent = list()

    for t in range(max_len):
        if len(set(paths[:, t])) < n_ag:
            collide = True
            c1 = Counter(paths[:, t])
            c2 = Counter(set(paths[:, t]))
            collision_node_idx = c1 - c2
            collision_node = list(collision_node_idx.elements())[0]
            collision_agent.append(tuple([i for i, n in enumerate(paths[:, t]) if n == collision_node]))

    return collide, collision_agent


def route_relatedness(routes, select_other):
    div = dict()
    for i, j in combinations(range(len(routes)), 2):  # route must include agent & task sequence's total coords
        dist = 0
        for a in routes[i]:
            for b in routes[j]:
                dist += (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))  # euclidean? A_star? manhattan?
        div[(i, j)] = dist / (len(routes[i]) * len(routes[j]))
    div = sorted(div.items(), key=lambda x: x[1])
    rd = np.random.choice([1, 2, 3])  # assume that agent number is bigger than 3 (3C2=3, 4C2=6)
    return (div, div[0][0]) if not select_other else (div, div[rd][0])
