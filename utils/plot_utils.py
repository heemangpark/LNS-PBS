import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def comparing_plot(plot_num):
    data_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'eval_')
    N_gap, heu_gap, rand_gap = 0, 0, 0
    for pn in range(plot_num):
        with open(data_dir + '{}.pkl'.format(pn), 'rb') as f:
            data = pickle.load(f)
        if (data[0] == 'stop') or (data[1] == 'stop'):
            pass
        else:
            plt.plot(np.arange(len(data[0])), data[0], label='SL_LNS')
            plt.plot(np.arange(len(data[1])), data[1], label='LNS')
            plt.xlabel('iteration'), plt.ylabel('route length')
            plt.legend(loc='upper right')
            plt.savefig('fig_{}.png'.format(pn))
            plt.clf()

            N_gap += (data[0][0] - data[0][-1]) / data[0][0] * 100
            rand_gap += (data[1][0] - data[1][-1]) / data[1][0] * 100

    print('SL_LNS: {} | LNS: {}'.format(N_gap / plot_num, rand_gap / plot_num))
