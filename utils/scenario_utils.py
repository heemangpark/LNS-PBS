import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.graph_utils import validGraph

curr_path = os.path.realpath(__file__)
scenario_dir = os.path.join(Path(curr_path).parent.parent, 'scenarios')

"""
1. Create random grid graph (user defined size, obstacle ratio)
2. Initialize the predefined positions of the agents and tasks
3. Save grid, graph, initial agent positions, task
"""


def save_scenarios(itrs=100, size=32, obs=20, T=1, a=10, t=20):
    instance, graph = validGraph(size, obs)

    for itr in range(9957, itrs):

        empty_idx = list(range(len(graph)))
        agent_idx = random.sample(empty_idx, a)
        tasks_len = [1 for _ in range(t)] if T == 1 else random.choices(list(range(1, T + 1)), k=t)
        agent_pos = np.array([a for a in graph])[agent_idx]
        empty_idx = list(set(empty_idx) - set(agent_idx))

        tasks = list()
        for i in range(t):
            temp_idx = random.sample(empty_idx, tasks_len[i])
            empty_idx = list(set(empty_idx) - set(temp_idx))
            tasks.append(np.array([t for t in graph])[temp_idx].tolist())

        datas = [instance, graph, agent_pos, tasks]
        dir = scenario_dir + '/{}{}{}_{}_{}_extra/'.format(size, size, obs, a, t)

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")

        with open(dir + 'scenario_{}.pkl'.format(itr), 'wb') as f:
            for d in datas:
                pickle.dump(d, f)


def load_scenarios(dir):
    dir = scenario_dir + '/' + dir
    data_list = []
    with open(dir, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
            data_list.append(data)

    return data_list


if __name__ == "__main__":
    save_scenarios(itrs=10000, size=32, obs=20, T=1, a=5, t=50)
