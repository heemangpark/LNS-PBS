import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from heuristics.regret import f_ijk, get_regret
from utils.graph_utils import sch_to_nx
from heuristics.hungarian import hungarian
from utils.solver_utils import to_solver, solver


class env_LNS:
    def __init__(self):
        self.grid = None
        self.ag_loc = np.array([])
        self.task_loc = np.array([])
        self.assign_id = []
        self.assign = {}
        self.cost = 0
        self.dirs = []

    def reset(self, scenario):
        env_dirs = [
            os.path.join(Path(os.path.realpath(__file__)).parent, '../EECBS/eecbs'),
            os.path.join(Path(os.path.realpath(__file__)).parent, 'EECBS/env/'),
            '',
            'env'
        ]
        try:
            if not os.path.exists(env_dirs[1]):
                os.makedirs(env_dirs[1])
        except OSError:
            print("Error: Cannot create the directory.")
        info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
        assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
        info['lns'] = assign_id, assign

        routes = to_solver(info['tasks'], assign)
        info['init_cost'], info['init_routes'] = solver(info['grid'], info['agents'], routes, dir=env_dirs)

        self.grid = info['graph']
        self.ag_loc = info['agents']
        self.task_loc = info['tasks']
        self.assign_id, self.assign = info['lns']
        self.cost = info['init_cost']

        schedules = self.ag_loc.tolist()
        for r in routes:
            schedules += r
        state = sch_to_nx(schedules, self.grid, len(self.ag_loc), len(self.task_loc))

        return state

    def step(self, action):
        done = False

        removal_idx = action
        for i, t in enumerate(self.assign.values()):
            for r in removal_idx:
                if {r: self.task_loc[r]} in t:
                    self.assign[i].remove({r: self.task_loc[r]})

        while len(removal_idx) != 0:
            f_val = f_ijk(self.assign, self.ag_loc, removal_idx, self.task_loc, self.grid)
            regret = get_regret(f_val)
            regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
            re_ins = list(regret.keys())[0]
            re_a, re_j = regret[re_ins][1], regret[re_ins][2]
            removal_idx.remove(re_ins)
            to_insert = {re_ins: self.task_loc[re_ins]}
            self.assign[re_a].insert(re_j, to_insert)

        cost, _ = solver(self.grid, self.ag_loc, to_solver(self.task_loc, self.assign), dir=self.dirs)

        if cost == 'error':
            state, reward, done = 'solver_error', 'solver_error', 'solver_error'

        else:
            reward = self.cost - cost if self.cost > cost else 0
            self.cost = cost

            routes = to_solver(self.task_loc, self.assign)
            schedules = self.ag_loc.tolist()
            for tr in routes:
                schedules += tr

            state = sch_to_nx(schedules, self.grid, len(self.ag_loc), len(self.task_loc))

        return state, reward, done
