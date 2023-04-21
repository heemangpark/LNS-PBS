import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def comparing_plot(plot_num):
    data_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'eval_')
    r_gap, b_gap, gap = 0, 0, 0
    for pn in range(plot_num):
        with open(data_dir + '{}.pkl'.format(pn), 'rb') as f:
            r_lns, b_lns, lns = pickle.load(f)
        if (r_lns == 'stop') or (b_lns == 'stop') or (lns == 'stop'):
            pass
        else:
            plt.plot(np.arange(len(r_lns)), r_lns, label='rLNS')
            plt.plot(np.arange(len(b_lns)), b_lns, label='bLNS')
            plt.plot(np.arange(len(lns)), lns, label='LNS')
            plt.xlabel('iteration'), plt.ylabel('route length')
            plt.legend(loc='upper right')
            plt.savefig('fig_{}.png'.format(pn))
            plt.clf()

            r_gap += (r_lns[0] - r_lns[-1]) / r_lns[0] * 100
            b_gap += (b_lns[0] - b_lns[-1]) / b_lns[0] * 100
            gap += (lns[0] - lns[-1]) / lns[0] * 100

    print('rLNS: {} || bLNS: {} || LNS: {}'.format(r_gap / plot_num, b_gap / plot_num, gap / plot_num))
