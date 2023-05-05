import bisect

import numpy as np


class Node:
    def __init__(self, parent=None, idx=None):
        self.parent = parent
        self.idx = idx
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.idx == other.idx

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h


def graph_astar(start, end, returnCostOnly=False, g=None):
    if start[0] == end[0] and start[1] == end[1]:
        return 0, 0
    start = tuple(start)
    end = tuple(end)
    path = list()
    path_cost = 0
    queue = list()
    openpath = dict()
    closepath = list()
    found = False
    node = Node(None, start)
    end_node = Node(None, end)

    openpath[start] = node
    queue.append(node)
    while openpath and not found:
        current_node = queue.pop(0)
        openpath.pop(current_node.idx)
        closepath.append(current_node)
        for new_idx in g.neighbors(current_node.idx):
            node = Node(current_node, new_idx)
            node.g = current_node.g + g.edges[current_node.idx, node.idx]['dist'].item()
            node.h = abs(np.array(g.nodes[node.idx]['loc']) - np.array(g.nodes[end_node.idx]['loc'])).sum().item()
            node.f = node.g + node.h

            if node in closepath:
                continue
            elif node == end_node:
                current = node
                while current is not None:
                    path.append(current.idx)
                    current = current.parent
                path = path[::-1]
                for p in range(len(path) - 1):
                    path_cost += g.edges[path[p], path[p + 1]]['dist'].item()
                if returnCostOnly:
                    return path_cost  # Return reversed path
                else:
                    return path, path_cost
            else:
                duplicated = openpath.get(node.idx)
                if not duplicated:
                    openpath[node.idx] = node
                    bisect.insort_left(queue, node)
                elif duplicated.g > node.g:
                    left = bisect.bisect_left(queue, duplicated)
                    right = bisect.bisect_right(queue, duplicated)
                    queue.pop(queue.index(duplicated, left, right))
                    openpath[node.idx] = node
                    bisect.insort_left(queue, node)
