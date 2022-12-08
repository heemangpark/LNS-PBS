import os.path
import pickle
import random

import numpy as np

from graph.generate_graph import graph

"""
1. Create random grid graph (user defined size, obstacle ratio)
2. Convert into graph
3. Initialize the predefined positions of the agents and tasks
4. Save grid, graph, initial agent positions, task
"""


def save_scenarios(itr=10, size=16, obs_ratio=0.1):
    for it in range(itr):

        # 1
        instance = np.zeros((size, size))
        obstacle = np.random.random((size, size)) <= obs_ratio
        instance[obstacle] = 1

        # 2
        g = graph(instance)
        # visualize(g)

        # 3
        C = 2
        num_agents = 5
        num_tasks = 10
        # empty_grid = (instance.reshape(-1) == 0).nonzero()[0].tolist()
        empty_idx = list(range(len(g)))
        agent_idx = random.sample(empty_idx, num_agents)
        tasks_len = random.choices(list(range(2, C + 1)), k=num_tasks)
        agent_pos = np.array([a for a in g])[agent_idx]
        empty_idx = list(set(empty_idx) - set(agent_idx))

        tasks = list()
        for i in range(num_tasks):
            temp_idx = random.sample(empty_idx, tasks_len[i])
            empty_idx = list(set(empty_idx) - set(temp_idx))
            tasks.append(np.array([t for t in g])[temp_idx].tolist())

        # 4
        datas = [instance, g, agent_pos, tasks]
        dir = '../instance_scenarios/{}_{}_{}/'.format(size, size, obs_ratio)

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")

        with open(dir + 'scenario_{}.pkl'.format(it + 1), 'wb') as f:
            for d in datas:
                pickle.dump(d, f)


def load_scenarios(dir):
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
    save_scenarios(10, 16, .1)
    save_scenarios(10, 16, .2)
    save_scenarios(10, 32, .1)
    save_scenarios(10, 32, .2)
    save_scenarios(10, 64, .1)
    save_scenarios(10, 64, .2)