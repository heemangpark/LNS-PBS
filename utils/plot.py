import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def comparing_plot(plot_num):
    data_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, './destroy/eval_')
    gap, n_gap = 0, 0
    for pn in range(plot_num):
        with open(data_dir + '{}.pkl'.format(pn), 'rb') as f:
            lns, nlns, _ = pickle.load(f)

        plt.plot(np.arange(len(lns)), lns, label='LNS')
        plt.plot(np.arange(len(nlns)), nlns, label='NLNS')
        plt.xlabel('iteration'), plt.ylabel('route length')
        plt.legend(loc='upper right')
        plt.savefig('fig_{}.png'.format(pn))
        plt.clf()

        gap += (lns[0] - lns[-1]) / lns[0] * 100
        n_gap += (nlns[0] - nlns[-1]) / nlns[0] * 100

    print('LNS: {} || NLNS: {}'.format(gap / plot_num, n_gap / plot_num))
