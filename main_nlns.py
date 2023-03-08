"""
1. initial solution (hungarian)
2. destroy operator (policy network)
3. repair operator (solver, search-based LNS)
"""
import os
from pathlib import Path

import dgl
import numpy as np
import torch.optim

from LNS.hungarian import hungarian
from LNS.regret import f_ijk, get_regret
from NLNS.policy_gnn import Policy_gnn
from seq_solver import seq_solver
from utils.generate_scenarios import load_scenarios

curr_path = os.path.realpath(__file__)
solver_dir = os.path.join(Path(curr_path).parent, 'EECBS/')


def is_connected(edges):
    return edges.data['co'] == 1


def destroy(edges):
    return {'co': edges.data['co'] - 1}


np.random.seed(42)
neigh = 5
d_op = Policy_gnn()
# wandb.init(project='ETRI')

for epi in range(50000):
    # import base dataset
    map_seed = np.random.randint(1, 501)
    scen_seed = np.random.randint(1, 11)
    scenario = load_scenarios('323220_1_10_10_{}/scenario_{}.pkl'.format(map_seed, scen_seed))
    grid, graph, origin_agents, origin_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

    # generate bipartite graph (represents agent-task assignment)
    AG = 1
    TASK = 2
    n_ag = len(origin_agents)
    n_task = len(origin_tasks)

    src_ag = [i for i in range(n_ag) for _ in range(n_task)]
    dst_task = list(range(n_ag, n_ag + n_task)) * n_ag
    bi_g = dgl.graph((src_ag, dst_task))

    ag_type = torch.FloatTensor([AG] * n_ag)
    task_type = torch.FloatTensor([TASK] * n_task)
    bi_g.ndata['type'] = torch.cat([ag_type, task_type])
    bi_g.edata['co'] = torch.zeros(bi_g.num_edges())

    # generate initial solution and calculate its path costs
    assign, assign_loc = hungarian(graph, origin_agents, origin_tasks)

    s_in_tasks = [[] for _ in range(n_ag)]
    for a, t in assign_loc.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += origin_tasks[list(_t.keys())[0]]
            s_in_tasks[a] = __t

    init_cost, paths = seq_solver(grid, origin_agents, s_in_tasks, {'time_limit': 60, 'sub_op': 1.2})

    # update bipartite graph with an initial solution
    for a, t in assign.items():
        if type(t) == np.int64:
            bi_g.edges[a, n_ag + t].data['co'] = torch.tensor([1.])
        else:
            for _t in t:
                bi_g.edges[a, n_ag + _t].data['co'] = torch.tensor([1.])
    bi_g.ndata['loc'] = torch.cat([torch.Tensor(origin_agents), torch.Tensor(s_in_tasks).squeeze(1)])

    # (input) edge connection variables -> policy network -> (output) destroy decision
    prob = d_op.actor(bi_g).squeeze(-1)
    c_prob = prob.detach().numpy()
    if [i == j for i, j in zip(c_prob[:-1], c_prob[1:])] == [True for _ in range(len(c_prob) - 1)]:
        action_id = np.random.choice(10, neigh, replace=False)
    else:
        c = np.sort(c_prob)[::-1][neigh - 1].item()
        action_id = torch.argwhere(prob >= c).squeeze(-1).tolist()
    action = [bi_g.filter_edges(is_connected).tolist()[a_id] for a_id in action_id]

    # destroy bipartite graph (1 -> 0)
    bi_g.apply_edges(destroy, action)  # destroy function applied to edge[action]
    l_agents = [bi_g.find_edges(e_id)[0].item() for e_id in bi_g.filter_edges(is_connected)]
    l_tasks = [bi_g.find_edges(e_id)[1].item() - n_ag for e_id in bi_g.filter_edges(is_connected)]

    for a in assign_loc.keys():
        if a in l_agents:
            pass
        else:
            assign_loc[a] = []

    # repair (regret method) and calculate its path costs
    tb_rep = list(set(range(n_task)) - set(l_tasks))
    while len(tb_rep) != 0:
        f = f_ijk(assign_loc, origin_agents, tb_rep, origin_tasks, graph)
        regret = get_regret(f)
        regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
        re_ins = list(regret.keys())[0]
        re_a, re_j = regret[re_ins][1], regret[re_ins][2]
        tb_rep.remove(re_ins)
        to_insert = {re_ins: origin_tasks[re_ins]}
        assign_loc[re_a].insert(re_j, to_insert)

    s_in_tasks = [[] for _ in range(n_ag)]
    for a, t in assign_loc.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += origin_tasks[list(_t.keys())[0]]
            s_in_tasks[a] = __t

    rep_cost, paths = seq_solver(grid, origin_agents, s_in_tasks, {'time_limit': 1, 'sub_op': 1.2})

    logit_sum = (prob[action_id]).log().sum()

    loss = (rep_cost - init_cost) * logit_sum

    d_op.optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(d_op.parameters(), 0.5)
    d_op.optim.step()

    if (epi + 1) % 1000 == 0:
        torch.save(d_op, os.path.join(Path(curr_path).parent.parent, 'LNS-PBS/NLNS/model_{}.th'.format(epi + 1)))

    # wandb.log({'loss': loss.item()})
