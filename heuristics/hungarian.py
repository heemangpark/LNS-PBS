import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.astar import graph_astar


def manhattan(coord_1, coord_2):
    x = abs(list(coord_1)[0] - list(coord_2)[0])
    y = abs(list(coord_1)[1] - list(coord_2)[1])
    return x + y


def cost_matrix(g, a, t):
    m = np.zeros((len(a), len(t)))
    for i in range(len(a)):
        for j in range(len(t)):
            # m[i][j] = graph_astar(g, a[i], t[j][0])[1]
            m[i][j] = manhattan(a[i], t[j][0])
    return m


def hungarian(graph, ag_pos_initial, task_pos):
    cm_initial = cost_matrix(graph, ag_pos_initial, task_pos)
    ag, assignment = linear_sum_assignment(cm_initial)
    list_assignment = [[a] for a in assignment]
    ret_dict = dict(zip(ag, list_assignment))
    tasks_idx = list(range(len(task_pos)))
    unassigned_idx = list(set(tasks_idx) - set(assignment))

    while len(unassigned_idx) != 0:
        ag_pos = [task_pos[t[-1]][-1] for t in ret_dict.values()]
        unassigned_pos = [task_pos[idx] for idx in unassigned_idx]
        cm = cost_matrix(graph, ag_pos, unassigned_pos)
        ag, assignment = linear_sum_assignment(cm)

        # update index
        assignment = [unassigned_idx[t_idx] for t_idx in assignment]
        # update unassigned idx
        unassigned_idx = list(set(unassigned_idx) - set(assignment))
        # append to ret dict
        for a_idx, t_idx in zip(ag, assignment):
            ret_dict[a_idx].append(t_idx)

    h_tasks = dict()
    for k in ret_dict.keys():
        # h_tasks[k] = [{'s': [agent_pos[k].tolist()]}]
        if type(list(ret_dict.values())[k]) == np.int64:
            i = list(ret_dict.values())[k]
            h_tasks[k] = [{i: task_pos[i]}]
        else:
            h_tasks[k] = [{i: task_pos[i]} for i in list(ret_dict.values())[k]]

    return ret_dict, h_tasks
