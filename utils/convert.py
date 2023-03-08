from collections import Counter


def to_solver(task_in_seq, assignment):
    s_in_tasks = [[] for _ in range(len(assignment))]
    for a, t in assignment.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += task_in_seq[list(_t.keys())[0]]
            s_in_tasks[a] = __t
    return s_in_tasks


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
