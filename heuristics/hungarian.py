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

def manhattan_dist(xs, ys):
    xs = xs.reshape((-1, 1, 2))
    ys = ys.reshape((1, -1, 2))
    matrix = np.abs(xs - ys).sum(-1)
    return matrix


def hungarian_prev(graph, ag_pos_initial, task_pos):
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


def hungarian(graph, ag_pos_initial, task_pos):
    n_ag = len(ag_pos_initial)
    ag_pos_initial = np.array(ag_pos_initial)
    task_pos = np.array(task_pos)
    ret_assignments = [[] for _ in range(n_ag)]
    ret_pos = [[] for _ in range(n_ag)]

    cm_initial = manhattan_dist(ag_pos_initial, task_pos)

    ag, assignment = linear_sum_assignment(cm_initial)
    for a, t in zip(ag, assignment):
        ret_assignments[a].append(t)
        ret_pos[a].append(task_pos[t])
    tasks_idx = np.arange((len(task_pos)))
    unassigned_idx = np.array(list(set(tasks_idx) - set(assignment)))

    while len(unassigned_idx) != 0:
        last_schedule_idx = [schedule[-1] for schedule in ret_assignments]
        ag_pos = task_pos[last_schedule_idx]
        unassigned_pos = task_pos[unassigned_idx]
        cm = manhattan_dist(ag_pos, unassigned_pos)
        ag, assignment = linear_sum_assignment(cm)

        # update index
        assignment = unassigned_idx[assignment]
        # update unassigned idx
        unassigned_idx = np.array(list(set(unassigned_idx) - set(assignment)))
        # append to ret dict
        for a, t in zip(ag, assignment):
            ret_assignments[a].append(t)
            ret_pos[a].append(task_pos[t])

    return ret_assignments, ret_pos
