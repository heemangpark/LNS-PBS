import torch
from main_nn import run_episode, Agent
import numpy as np

if __name__ == '__main__':
    heuristic = False
    VISUALIZE = True
    M, N = 10, 50
    T_threshold = 10  # N step fwd
    n_sample = 100
    agent = Agent()
    agent.load_state_dict(torch.load('saved/20230221_0902.th'))
    # agent.load_state_dict(torch.load('saved/20230220_1424.th'))
    exp_name = 'test'
    # eval_i = 2

    episode_timesteps = []
    for eval_i in range(20):
        scenario_dir = '323220_{}_{}/scenario_{}.pkl'.format(10, 50, 11)
        episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=True,
                                            scenario_dir=scenario_dir, VISUALIZE=VISUALIZE, heuristic=heuristic,
                                            n_sample=n_sample)
        # print(episode_timestep)
        if episode_timestep is not None:
            episode_timesteps.append(episode_timestep)
        print(np.mean(episode_timesteps))

