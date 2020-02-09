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

import argparse

def plot(ax,input_type,adaptation_mode):

    file = get_simfile_prop(os.path.join(DATA_DIR,input_type
    +'_input_ESN/alt_hom_regulation/alt_hom_regulation_'+adaptation_mode))

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

    parser = argparse.ArgumentParser()

    parser.add_argument("input_type",
    help='''specify four type of input (homogeneous_identical_binary,
    homogeneous_independent_gaussian, heterogeneous_identical_binary,
    heterogeneous_independent_gaussian)''',
    choices=['homogeneous_identical_binary',
    'homogeneous_independent_gaussian',
    'heterogeneous_identical_binary',
    'heterogeneous_independent_gaussian'])

    parser.add_argument("adaptation_mode",
    help='''specify the mode of adaptation: local or global''',
    choices=['local','global'])

    args = parser.parse_args()


    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*0.8,TEXT_WIDTH*0.6))

    plot(ax,args.input_type,args.adaptation_mode)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_alt_hom_regulation_check_conv_'
    +args.adaptation_mode
    +'.pdf'))
    
    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_alt_hom_regulation_check_conv_'
    +args.adaptation_mode
    +'.png'),dpi=1000)

    plt.show()
