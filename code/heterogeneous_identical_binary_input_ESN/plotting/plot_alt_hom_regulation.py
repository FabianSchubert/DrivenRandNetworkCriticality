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
    print(sigm_w_e)
    eps_a = dat['eps_a']
    print(eps_a)
    eps_b = dat['eps_b']
    mu_y_target = dat['mu_y_target']

    n_averaging_inp = 1000

    a = np.linspace(0.,2.5,500)
    vy = np.linspace(0.,1.*N,500)

    A,VY = np.meshgrid(a,vy)

    delta_a = eps_a*A*(1.-A**2.)*VY/N

    delta_vy = np.zeros((500,500))

    for k in tqdm(range(n_averaging_inp)):

        delta_vy += N*(1-(1. + 2*A**2.*VY/N + 2.*np.random.normal(0.,sigm_w_e)**2.)**(-.5))-VY

    delta_vy /= n_averaging_inp


    ax.streamplot(A,VY,delta_a,delta_vy)

    vy_pl = np.linspace(0.,1.*N,1000)
    a_pl = np.linspace(0.,2.5,1000)



    #ax.plot((((1.-vy_pl/N)**(-2.)/2. - v_e - .5 )/(sigm_w**2.*vy_pl/N))**.5,vy_pl)
    #ax.plot(0.*a + sigm_w**(-1.),vy)

    for k in range(50):
        plt.plot(a_rec[k,:,0],y_norm_rec[k,:]**2.,c=colors[1],alpha=1.,lw=1)
        #plt.plot(a_rec[k,1:,0])
        #plt.plot(y_norm_rec[k,1:])

    ax.contour(a,vy,delta_a,levels=[0.],colors=[colors[2]],linewidths=[2.],zorder=3)
    ax.contour(a,vy,delta_vy,levels=[0.],colors=[colors[3]],linewidths=[2.],zorder=4)

    ax.set_xlim([a_pl[0]-.1,a_pl[-1]])
    ax.set_ylim([vy_pl[0]-30.,vy_pl[-1]])

    ax.set_xlabel('$a$')
    ax.set_ylabel('$||\\mathbf{y}||^2$')

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_alt_hom_regulation.png'),dpi=1000)

    plt.show()
