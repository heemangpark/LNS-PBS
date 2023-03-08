import csv

import numpy as np
from tqdm import trange

from LNS.hungarian import hungarian
from utils.cross_exchange import CE_relatedness
from utils.generate_scenarios import load_scenarios
from utils.lns import LNS
from utils.solver_util import to_solver
from seq_solver import seq_solver
np.random.seed(42)
exp_length = 100
report = [[], []]

for itr in trange(1, exp_length + 1):
    scenario = load_scenarios('202020_1_5_50_0/scenario_{}.pkl'.format(itr))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    assign_id, assign = hungarian(info['graph'], info['agents'], info['tasks'])
    info['ce_assign'] = to_solver(info['tasks'], assign)
    info['lns_assign'] = (assign_id, assign)
    info['init_cost'], _ = seq_solver(info['grid'], info['agents'], info['ce_assign'], {'time_limit': 1, 'sub_op': 1.2})
    if info['init_cost'] == 'error':
        pass
    else:
        ce, lns = CE_relatedness(info), LNS(info)
        if (ce == 'NaN') or (lns == 'NaN'):
            pass
        else:
            report[0].append(ce), report[1].append(lns)

with open('exp_rela.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(report)
