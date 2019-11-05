#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os

from src.analysis_tools import get_simfile_prop

def plot(ax):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cmap = mpl.cm.get_cmap('viridis')

    simfile,timestamp = get_simfile_prop(os.path.join(DATA_DIR,'heterogeneous_independent_gaussian_input_ESN/param_sweep_'))

    dat = np.load(simfile)

    sigm_t = dat['sigm_t']
    sigm_e = dat['sigm_e']

    n_sigm_t = sigm_t.shape[0]
    n_sigm_e = sigm_e.shape[0]

    y = dat['y']

    N = y.shape[3]

    corr = np.ndarray((n_sigm_e,n_sigm_t,N,N))
    avg_off_diag = np.ndarray((n_sigm_e,n_sigm_t))
    avg_diag = np.ndarray((n_sigm_e,n_sigm_t))

    for k in range(n_sigm_e):
        for l in range(n_sigm_t):
            corr[k,l,:,:] = np.corrcoef(y[k,l,:,:].T)
            avg_off_diag[k,l] = (np.abs(corr[k,l,:,:]).sum() - np.abs(corr[k,l,range(N),range(N)]).sum())/(N**2-N)
            avg_diag[k,l] = np.abs(corr[k,l,range(N),range(N)]).mean()




    for k in range(n_sigm_e):
        col = cmap(0.8*k/n_sigm_e)
        ax.plot(sigm_t,avg_off_diag[k,:],color=col,label='$\\sigma_{\\rm ext} = $' +  str(sigm_e[k]))

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('Mean Activity Cross Correlation')

    ax.legend()

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_independent_gaussian_input_corr_act.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_independent_gaussian_input_corr_act.png'),dpi=1000)

    plt.show()
