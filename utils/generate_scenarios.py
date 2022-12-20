import os
import os.path
import pickle
import random
from pathlib import Path

import numpy as np

from graph.generate_graph import gen_graph
from utils.vis_graph import vis_graph

curr_path = os.path.realpath(__file__)
scenario_dir = os.path.join(Path(curr_path).parent.parent, 'scenarios')

"""
1. Create random grid graph (user defined size, obstacle ratio)
2. Initialize the predefined positions of the agents and tasks
3. Save grid, graph, initial agent positions, task
"""


def save_scenarios(itr=10, size=32, obs=20, C=1, M=10, N=10):
    """
    C: task length -> if 2, tau=(s, g)
    M: the number of agents
    N: the number of tasks
    """

    # 1
    instance, graph = gen_graph(size, obs)
    # vis_graph(graph)

    for it in range(itr):

        # 3
        empty_idx = list(range(len(graph)))
        agent_idx = random.sample(empty_idx, M)
        if (C == 1) or (C == 2):
            tasks_len = [C for _ in range(N)]
        else:
            tasks_len = random.choices(list(range(1, C + 1)), k=N)
        agent_pos = np.array([a for a in graph])[agent_idx]
        empty_idx = list(set(empty_idx) - set(agent_idx))

        tasks = list()
        for i in range(N):
            temp_idx = random.sample(empty_idx, tasks_len[i])
            empty_idx = list(set(empty_idx) - set(temp_idx))
            tasks.append(np.array([t for t in graph])[temp_idx].tolist())

        # 4
        datas = [instance, graph, agent_pos, tasks]
        dir = scenario_dir + '/{}{}{}_{}_{}_{}/'.format(size, size, obs, C, M, N)

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")

        with open(dir + 'scenario_{}.pkl'.format(it + 1), 'wb') as f:
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

# if __name__ == "__main__":
#     save_scenarios()
