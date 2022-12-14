import subprocess

from utils.generate_scenarios import save_scenarios, load_scenarios
from utils.solver_util import save_map, save_scenario

C, M, N = 1, 10, 10
save_scenarios(C=C, M=M, N=N)
scenario = load_scenarios('323220_{}_{}_{}/scenario_4.pkl'.format(C, M, N))
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]

scenario_name = 'test'
save_map(grid, scenario_name)
save_scenario(agent_pos, total_tasks, scenario_name, grid.shape[0], grid.shape[1])

solver_path = "EECBS/"
c = [solver_path + "eecbs",
     # "-m", solver_path + "random-32-32-20.map",
     "-m", solver_path + scenario_name + '.map',
     # "-a", solver_path + "random-32-32-20-random-1.scen",
     "-a", solver_path + scenario_name + '.scen',
     "-o", solver_path + "test.csv",
     "--outputPaths = EECBS/paths.txt",
     "-k", "{}".format(M),
     "-t", "60",
     "--suboptimality = 1.2"]

subprocess.run(c)

# def invoke_EECBS(cmd):
#     subprocess.Popen(cmd)
#
#
# if __name__ == '__main__':
#     cmd = "./eecbs -m random-32-32-20.map -a random-32-32-20-random-1.scen -o test.csv --outputPaths=paths.txt -k 50 -t 60 --suboptimality=1.2 "
#     print('A')
#     invoke_EECBS(cmd)
