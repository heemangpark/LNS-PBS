import csv

from tqdm import trange

from main_ce import CE_length, CE_random
from main_lns import LNS
from utils.generate_scenarios import load_scenarios

report = [[], []]
for itr in trange(1, 101):
    scenario = load_scenarios('101020_1_10_30_0/scenario_{}.pkl'.format(itr))
    info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2], 'tasks': scenario[3]}
    report[0].append(LNS(info)), report[1].append(CE_random(info))

with open('exp_LNS_rand.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(report)
