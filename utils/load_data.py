import os
from pathlib import Path

import pickle

curr_path = os.path.realpath(__file__)
data_dir = os.path.join(Path(curr_path).parent.parent, 'data/')


def load_oracle():
    name = '1010'
    with open(data_dir + name + 'AP_' + '1', 'rb') as f:
        ag_pos = pickle.load(f)
    with open(data_dir + name + 'grid_' + '1', 'rb') as f:
        grid = pickle.load(f)
    with open(data_dir + name + 'TA_' + '1', 'rb') as f:
        tasks = pickle.load(f)
    with open(data_dir + name + 'graph_' + '1', 'rb') as f:
        graph = pickle.load(f)

    return grid, graph, ag_pos, tasks


if __name__ == '__main__':
    grid, graph, ag_pos, tasks = load_oracle()
