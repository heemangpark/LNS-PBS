import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.astar import graph_astar


def cost_matrix(g, ap, t):
    m = np.zeros((len(ap), len(t)))
    for i in range(len(ap)):
        for j in range(len(t)):
            m[i][j] = graph_astar(g, ap[i], t[j][0])[1]

    return m


def hungarian(graph, agent_pos, tasks):
    cm = cost_matrix(graph, agent_pos, tasks)
    ag, assigned = linear_sum_assignment(cm)
    task_idx = dict(zip(ag, assigned))
    tasks_idx = list(range(len(tasks)))
    unassigned_idx = list(set(tasks_idx) - set(assigned))
    unassigned = [tasks[ut] for ut in unassigned_idx]

    first = True
    while len(unassigned) != 0:
        if first:
            na = [tasks[t][-1] for t in task_idx.values()]
        else:
            na = [tasks[t[-1]][-1] for t in task_idx.values()]
        cm = cost_matrix(graph, na, unassigned)
        ag, assigned = linear_sum_assignment(cm)
        assigned = [unassigned_idx[t_idx] for t_idx in assigned]
        unassigned_idx = list(set(unassigned_idx) - set(assigned))
        for a, t in zip(ag, assigned):
            if type(task_idx[a]) == np.int64:
                task_idx[a] = [task_idx[a]] + [t]
                unassigned.remove(tasks[t])
            else:
                task_idx[a].append(t)
                unassigned.remove(tasks[t])
        first = False

    h_tasks = dict()
    for k in task_idx.keys():
        # h_tasks[k] = [{'s': [agent_pos[k].tolist()]}]
        if type(list(task_idx.values())[k]) == np.int64:
            i = list(task_idx.values())[k]
            h_tasks[k] = [{i: tasks[i]}]
        else:
            h_tasks[k] = [{i: tasks[i]} for i in list(task_idx.values())[k]]

    return task_idx, h_tasks
