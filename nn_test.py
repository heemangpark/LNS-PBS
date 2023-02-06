import torch
from main_nn import run_episode, Agent
import numpy as np

if __name__ == '__main__':
    heuristic = False
    VISUALIZE = False
    M, N = 10, 20
    T_threshold = 10  # N step fwd
    n_sample = 5
    agent = Agent()
    agent.load_state_dict(torch.load('saved/20230203_1016.th'))
    exp_name = 'test'
    # eval_i = 2

    episode_timesteps = []
    for eval_i in range(20):
        scenario_dir = '323220_1_{}_{}_eval/scenario_{}.pkl'.format(M, N, eval_i + 1)
        episode_timestep, itr = run_episode(agent, M, N, exp_name, T_threshold, sample=False,
                                            scenario_dir=scenario_dir, VISUALIZE=VISUALIZE, heuristic=heuristic,
                                            n_sample=n_sample)
        # print(episode_timestep)
        if episode_timestep is not None:
            episode_timesteps.append(episode_timestep)

    print(np.mean(episode_timesteps))
