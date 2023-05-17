import numpy as np
from scipy.optimize import linear_sum_assignment


def manhattan_dist(xs, ys):
    xs = xs.reshape((-1, 1, 2))
    ys = ys.reshape((1, -1, 2))
    matrix = np.abs(xs - ys).sum(-1)
    return matrix


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
