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

    r_a = a_rec[0,:,:]**2. * (W**2.).sum(axis=1)

    ax.plot(r_a[:,0],c=colors[0],alpha=0.25,label='$R_{{\\rm a},i}$')
    ax.plot(r_a[:,1:],c=colors[0],alpha=0.25)

    ax.plot(r_a.mean(axis=1),'--',c='k',label='$\\left\\langle R_{{\\rm a},i} \\right\\rangle_{\\rm P}$',lw=2)

    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    ax.set_xlabel('time steps')
    ax.set_ylabel('$R_{{\\rm a},i}$')

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*0.8,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation_r_a.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation_r_a.png'),dpi=1000)

    plt.show()
