import torch
from main_nn import run_episode, Agent

if __name__ == '__main__':
    M, N = 10, 20
    T_threshold = 10  # N step fwd
    agent = Agent()
    agent.load_state_dict(torch.load('saved/20230201_1611.th'))
    exp_name = 'test'
    eval_i = 4

    VISUALIZE = True
    scenario_dir = '323220_1_{}_{}_eval/scenario_{}.pkl'.format(M, N, eval_i + 1)
    episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=False,
                                        scenario_dir=scenario_dir, VISUALIZE=VISUALIZE)
    print(episode_timestep)