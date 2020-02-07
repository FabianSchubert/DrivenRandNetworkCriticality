#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop
import scipy.signal as sg

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

def plot(ax):

    file = get_simfile_prop(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/alt_hom_regulation'))

    dat = np.load(file[0])

    a_rec = dat['a']
    y_norm = dat['y_norm'][0,:]
    y=dat['y']
    N=dat['N']
    n_samples=dat['n_samples']
    cf_w = dat['cf_w']
    cf_w_in = dat['cf_w_in']
    sigm_w_e = dat['sigm_w_e']
    eps_a = dat['eps_a']
    eps_b = dat['eps_b']
    mu_y_target = dat['mu_y_target']
    W = dat['W']
    X_r = dat['X_r']

    h = sg.get_window('triang',500)
    for k in range(100):
        filt_sign = sg.convolve(y[:-1,k]**2.-X_r[1:,k]**2.,h/h.sum(),mode='same')
        plt.plot(filt_sign)

    #ax.plot(y_norm**2./N)
    #ax.set_xlim([0.,.5])
    #ax.set_ylim([0.,.5])

    ax.set_xlabel('time steps')
    ax.set_ylabel('$y_i^2(t-1) - X_{{\\rm r},i}^2(t)$ (Trailing Average)')

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*0.8,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_independent_gaussian_input_alt_hom_regulation_check_conv.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_independent_gaussian_input_alt_hom_regulation_check_conv.png'),dpi=1000)

    plt.show()
