import heapq as hq
from math import sqrt


def euclidean_dist(node1_x, node1_y, node2_x, node2_y):
    return sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)


class Node():
    def __init__(self, x, y, l, g, task, t, parent=-1) -> None:
        self.x = x
        self.y = y
        self.l = l
        self.g = g
        self.t = t  # start node is at t=0
        self.task = task
        self.compute_h()
        self.f = self.g + self.h
        self.parent = parent
        # self.p = [self.x, self.y]

    def compute_h(self):
        if self.l == 1:
            assert len(self.task) == 2
            self.h = euclidean_dist(self.x, self.y, self.task[0][0], self.task[0][1]) + \
                     euclidean_dist(self.task[0][0], self.task[0][1], self.task[1][0], self.task[1][1])
        else:
            assert self.l == 2
            assert len(self.task) == 1
            self.h = euclidean_dist(self.x, self.y, self.task[0][0], self.task[0][1])

    def __str__(self):
        return 'x:{} y:{} l:{} t:{} g:{} h:{} task:{}'.format(self.x, self.y, self.l, self.t, self.g, self.h, self.task)


class Path():
    def __init__(self) -> None:
        self.xs = []
        self.ys = []
        self.ts = []
        self.len = 0

    def add_node(self, x, y):
        self.xs.insert(0, x)
        self.ys.insert(0, y)
        self.len += 1

    def add_time(self):
        self.ts = [i for i in range(self.len)]

    def __str__(self):
        print_str = ''
        assert len(self.xs) == len(self.ys) == len(self.ts), '{} {} {}'.format(len(self.xs), len(self.ys), len(self.ts))
        for i in range(len(self.ys)):
            print_str += '{} {} {} -> '.format(self.xs[i], self.ys[i], self.ts[i])
        return print_str


def collision_exists(curr_x, curr_y, curr_t, prev_x, prev_y, prev_t, paths):
    for other_ag_path in paths:
        if len(other_ag_path.xs) > curr_t:
            if curr_x == other_ag_path.xs[curr_t] and curr_y == other_ag_path.ys[curr_t]:
                return True
            elif curr_x == other_ag_path.xs[prev_t] and curr_y == other_ag_path.ys[prev_t] and \
                    prev_x == other_ag_path.xs[curr_t] and prev_y == other_ag_path.ys[curr_t]:
                return True
    return False


def generate_children(parent, nav_space, paths, nodes, Q, num_generated):
    if debug:
        print('children:')
    for child_x in [parent.x - 1, parent.x, parent.x + 1]:
        for child_y in [parent.y - 1, parent.y, parent.y + 1]:
            if child_x == parent.x and child_y == parent.y:
                continue
            if child_x < 0 or child_x >= nav_space.shape[1]:
                continue
            if child_y < 0 or child_y >= nav_space.shape[0]:
                continue
            if nav_space[nav_space.shape[0] - 1 - child_y, child_x] == 0:  # if there's a static obstacle
                continue
            child_t = parent.t + 1
            if collision_exists(child_x, child_y, child_t, parent.x, parent.y, parent.t, paths):
                continue

            additional_g = 1
            child_g = parent.g + additional_g
            child_l = parent.l
            child_task = parent.task
            child_t = parent.t + 1
            child = Node(child_x, child_y, child_l, child_g, child_task, child_t, parent)
            if debug:
                print(child)
            if (child_x, child_y, child_l) not in nodes:
                num_generated += 1
                hq.heappush(Q, (child.f, num_generated, child))
                nodes[(child_x, child_y, child_l)] = child
            else:
                prev_cost = nodes[(child_x, child_y, child_l)].f
                if child.f < prev_cost:
                    nodes[(child_x, child_y, child_l)] = child
                    for idx in range(len(Q)):
                        if Q[idx][2].x == child_x and Q[idx][2].y == child_y and Q[idx][2].l == child_l:
                            Q[idx] = (child.f, Q[idx][1], child)
                            hq.heapify(Q)
                            break
    return Q, num_generated


def backtrack(final_node):
    path = Path()
    curr_node = final_node
    while not isinstance(curr_node, int):
        path.add_node(curr_node.x, curr_node.y)
        curr_node = curr_node.parent
    path.add_time()
    return path


def mla_star(start, task, paths, nav_space):  # returns a path object
    Q = []
    nodes = {(start.x, start.y, start.l): start}
    num_generated = 1
    hq.heappush(Q, (start.f, num_generated, start))
    hq.heapify(Q)
    while len(Q) != 0:
        _, _, curr_node = hq.heappop(Q)
        if debug:
            print('*******************************\nparent:')
            print(curr_node)
        if curr_node.l == 1:
            assert len(curr_node.task) == 2
            pick_x = curr_node.task[0][0]
            pick_y = curr_node.task[0][1]
            t_max = 1e5
            for other_ag_path in paths:
                for i in range(other_ag_path.len):
                    if other_ag_path.xs[i] == pick_x and other_ag_path.ys[i] == pick_y:
                        t_max = other_ag_path.ts[i]
                        break
                if t_max != 1e5:
                    break
            if curr_node.g > t_max:
                if debug:
                    print('T_max exceeded. Skipping expansion.')
                continue
        if curr_node.l == 1:
            if curr_node.x == curr_node.task[0][0] and curr_node.y == curr_node.task[0][1]:
                new_node = Node(curr_node.x, curr_node.y, 2, curr_node.g, [curr_node.task[1]], curr_node.t, curr_node)
                num_generated += 1
                hq.heappush(Q, (new_node.f, num_generated, new_node))
                nodes[(new_node.x, new_node.y, new_node.l)] = new_node
                if debug:
                    print('Reached pickup location. Skipping expansion.')
                continue
        elif curr_node.l == 2:
            assert len(curr_node.task) == 1, '{}'.format(curr_node.task)
            if curr_node.x == curr_node.task[0][0] and curr_node.y == curr_node.task[0][1]:
                path = backtrack(curr_node)
                return path

        Q, num_generated = generate_children(curr_node, nav_space, paths, nodes, Q, num_generated)

    return None


def get_start_locations(num_agents):  # outputs in (x_coordinate, y_coordinate) format
    start_locs = [[1, 10], [1, 15], [1, 19]]
    # start_locs = start_locs[::-1]
    return start_locs


def get_tasks(num_agents):  # outputs in (x_coordinate, y_coordinate) format
    tasks = [[[4, 10], [0, 10]], [[4, 15], [0, 15]], [[4, 19], [0, 19]]]
    # tasks = tasks[::-1]
    return tasks


def sane_starts_tasks(starts, tasks, nav_space, num_agents):
    assert len(starts) == len(tasks) == num_agents, '{} {} {}'.format(len(starts), len(tasks), num_agents)
    for start in starts:
        assert 0 <= start[0] < nav_space.shape[1], '{} {}'.format(start[0], nav_space.shape[1])
        assert 0 <= start[1] < nav_space.shape[0], '{} {}'.format(start[1], nav_space.shape[0])
    for task in tasks:
        # assert len(task) == 2
        for i in range(len(task)):
            # for i in range(2):
            assert 0 <= task[i][0] < nav_space.shape[1], '{} {}'.format(task[i][0], nav_space.shape[1])
            assert 0 <= task[i][1] < nav_space.shape[0], '{} {}'.format(task[i][1], nav_space.shape[0])


if __name__ == '__main__':
    from utils.generate_scenarios import load_scenarios

    scen = load_scenarios('202020_1_5_50_0/scenario_1.pkl')

    debug = 0
    num_agents = len(scen[2])
    starts = [[3, 11], [15, 11], [0, 19], [15, 14], [2, 14]]
    tasks = [[[3, 12], [3, 13], [3, 10], [3, 8], [5, 3], [4, 1], [1, 1], [12, 3], [16, 5], [14, 1]],
             [[14, 12], [13, 12], [12, 15], [11, 15], [14, 19], [18, 19], [18, 16], [1, 0], [17, 10], [17, 8]],
             [[1, 15], [0, 16], [4, 17], [7, 16], [9, 15], [16, 16], [18, 15], [13, 7], [18, 6], [18, 10]],
             [[15, 13], [11, 13], [10, 13], [10, 14], [8, 8], [9, 4], [10, 3], [18, 12], [16, 0], [11, 2]],
             [[2, 15], [0, 14], [2, 11], [0, 9], [7, 9], [12, 6], [14, 6], [18, 8], [19, 0], [11, 1]]]
    paths = []
    nav_space = scen[0]
    # nav_space is such that if it is displayed with cv2.imshow, it'll show the map as one would draw it
    # nav_space[0] corresponds to top-most row (corresponding to y=nav_space.shape[0]-1)
    # nav_space[:, 0] corresponds to left-most column (corresponding to x=0)
    # x varies from 0 to nav_space.shape[1]-1
    # y varies from 0 to nav_space.shape[0]-1
    sane_starts_tasks(starts, tasks, nav_space, num_agents)
    for i in range(num_agents):
        task = tasks[i]
        start = starts[i]
        start = Node(start[0], start[1], 1, 0, task, 0)
        path = mla_star(start, task, paths, nav_space)
        if path is None:
            print('No path found for agent-' + str(i))
            input('Press enter to continue')
        else:
            paths.append(path)
            print('Path of agent-' + str(i))
            print(path)
