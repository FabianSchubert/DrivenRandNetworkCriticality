#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

def plot(ax):

    file = get_simfile_prop(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/alt_hom_regulation'))

    dat = np.load(file[0])

    a_rec = dat['a']
    y_norm_rec = dat['y_norm']
    N=dat['N']
    n_samples=dat['n_samples']
    cf_w = dat['cf_w']
    cf_w_in = dat['cf_w_in']
    sigm_w_e = dat['sigm_w_e']
    eps_a = dat['eps_a']
    eps_b = dat['eps_b']
    mu_y_target = dat['mu_y_target']
    W = dat['W']

    l_start = np.linalg.eigvals((W.T * a_rec[0,0,:]).T)
    l_end = np.linalg.eigvals((W.T * a_rec[0,-1,:]).T)

    ax.plot(l_start.real,l_start.imag,'.',markersize=5,label='$t=0$')
    sc_not_exp = int(np.log10(a_rec.shape[1]))
    sc_not_fact = a_rec.shape[1]/10**sc_not_exp
    ax.plot(l_end.real,l_end.imag,'.',markersize=5,label='$t='+str(sc_not_fact)+'\\times 10^'+str(sc_not_exp)+'$')

    ax.set_xlabel('$\\mathrm{Re}(\\lambda_i)$')
    ax.set_ylabel('$\\mathrm{Im}(\\lambda_i)$')

    ax.legend()

    ax.axis('equal')

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*0.8,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation_specrad.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation_specrad.png'),dpi=1000)

    plt.show()
