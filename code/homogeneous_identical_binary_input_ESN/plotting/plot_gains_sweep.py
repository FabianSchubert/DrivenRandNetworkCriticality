#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

import scipy.integrate as integrate
from scipy.optimize import newton_krylov

from stdParams import *
import os,glob,sys,re

from tqdm import tqdm

from src.analysis_tools import get_simfile_prop

def plot(ax):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cmap = mpl.cm.get_cmap('viridis')

    simfile,timestamp = get_simfile_prop(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/gains_sweep/gains_sweep_run_'))

    print('Loading data from ' + simfile + '...')

    dat = np.load(simfile)

    sigm_t = dat['sigm_t']
    sigm_e = dat['sigm_e']

    n_sigm_t = sigm_t.shape[0]
    n_sigm_e = sigm_e.shape[0]

    a = dat['a']

    #a_mean = a.mean(axis=2).T
    #a_std = a.std(axis=2).T



    ### solve analytical approximation numerically
    a_pred = np.ndarray((n_sigm_e,n_sigm_t))

    W = dat['W']
    N = W.shape[2]

    a_norm = ((a**2.).sum(axis=2).T/N)**.5

    sigm_w = W.std(axis=3)*N**.5

    ### check if cache file (starting with 'gain_solutions...') with solutions exsists and has the right time stamp:
    cachefile_list = glob.glob(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/gains_sweep/gains_solutions_*'))

    cachefile_load = None

    timestamp_regex = re.compile('[\-T:\.0-9]+(?=\.np)')

    for cachefile in cachefile_list:
        timestamp_cache = timestamp_regex.findall(cachefile)[0]

        if timestamp_cache == timestamp:
            cachefile_load = cachefile
            break

    if cachefile_load != None:
        print('Loading gain solutions from cachefile')
        a_pred = np.load(cachefile_load)
    else:
        print('No matching cachefile found!')

        print('solving analytical gain approximation...')

        ### solve analytical approximation numerically
        a_pred = np.ndarray((n_sigm_e,n_sigm_t,N))

        X_e = dat['X_e']

        sigm_X_e = X_e.std(axis=2)

        for l in tqdm(range(n_sigm_t)):
            for k in tqdm(range(n_sigm_e)):
                if not(sigm_e[k]==0 and sigm_t[l]==0):
                    for n in tqdm(range(N)):
                        sigm_squ = sigm_w[k,l,n]**2.*sigm_t[l]**2. + sigm_X_e[k,l,n]**2.
                        a_sol_pred = (((1.-sigm_t[l]**2.)**(-2.) - 1.)/(2.*sigm_squ))**.5
                        func = lambda A: integrate.quad(lambda x: np.tanh(A*x)**2.*np.exp(-0.5*x**2./sigm_squ),-np.inf,np.inf)[0]/(2.*np.pi*sigm_squ)**.5 - sigm_t[l]**2.
                        solver_res = newton_krylov(func,a_sol_pred)
                        a_pred[k,l,n] = solver_res[()]
                else:
                    a_pred[k,l,:] = 1.
        print('saving result...')
        np.save(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/gains_sweep/gains_solutions_'+timestamp+'.npy'),a_pred)
    ####

    a_pred_norm = ((a_pred**2.).sum(axis=2)/N)**.5

    for k in range(n_sigm_e):

        col = cmap(0.8*k/n_sigm_e)
        #ax.plot(sigm_t,a_mean[:,k],'^',color=col,label='$\\sigma_{\\rm ext} = $' +  str(sigm_e[k]))
        #ax.fill_between(sigm_t,a_mean[:,k]-a_std[:,k],a_mean[:,k]+a_std[:,k],color=col,alpha=0.3)
        #ax.errorbar(sigm_t,a_mean[:,k],yerr=a_std[:,k],color=col,ecolor=col,linestyle='',marker='^',markersize=6,label='$\\sigma_{\\rm ext} = $' +  str(sigm_e[k]))

        ax.plot(sigm_t,a_norm[:,k],color=col,linestyle='',marker='^',markersize=6,label='$\\sigma_{\\rm ext} = $' +  str(sigm_e[k]))

        ax.plot(sigm_t,a_pred_norm[k,:],color=col)

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('$R_{\\rm a}$')

    ax.legend()

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_identical_binary_input_gains.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_identical_binary_input_gains.png'),dpi=1000)

    plt.show()
