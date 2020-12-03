#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *

from tqdm import tqdm

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
viridis = sns.color_palette("viridis", as_cmap=True)

def plot(ax,datafile,
        T_dr=10,
        dr_a_thresh=1e-3,
        r_a_thresh=0.25):
    
    data = np.load(datafile)
    
    r_t = data["r_t"]
    sigm_e = data["sigm_e"]
    a = data["a"]
    #W = data["W"]
    T_skip = data["T_skip"]
    
    r_a_est = data["r_a_est"]
    
    n_r_t = r_t.shape[0]
    n_sigm_e = sigm_e.shape[0]
    N = a.shape[-1]
    n_samples = a.shape[0]
    n_t = a.shape[1]
    
    '''
    r_a_est = np.ndarray((n_samples,n_t,n_sigm_e,n_r_t))
    
    for s in tqdm(range(n_samples)):
        for k in tqdm(range(n_sigm_e),leave=False):
            for l in tqdm(range(n_r_t),leave=False):
                W_var = W[s,k,l].var(axis=1)
                for t in tqdm(range(n_t),leave=False):
                    r_a_est[s,t,k,l] = (a[s,t,k,l]**2.*W_var).sum()**.5
    '''
    
    dr_a_est = (r_a_est[:,T_dr:] - r_a_est[:,:-T_dr])/T_dr
    
    T_conv = np.argmax(((np.abs(dr_a_est)<dr_a_thresh)*(np.abs(r_a_est[:,T_dr:]-r_t)<r_a_thresh)),axis=1)*T_skip
    
    T_conv_mask = np.ma.masked_array(T_conv,mask = np.abs(r_a_est[:,-1]-r_t)>r_a_thresh)
    
    r_t_pcm = np.append(r_t,2.*r_t[-1]-r_t[-2])-0.5*(r_t[1]-r_t[0])
    sigm_e_pcm = np.append(sigm_e,2.*sigm_e[-1]-sigm_e[-2])-0.5*(sigm_e[1]-sigm_e[0])
    
    T_conv_mean = T_conv_mask.mean(axis=0)
    T_conv_err = T_conv_mask.std(axis=0)#/T_conv_mask.shape[0]**.5
    
    k_max_plot = 8
    viridis_max = 0.95
    
    for k in range(1,k_max_plot+1):
        #if(r_t[k] in (r_t[1],0.5,1.,1.5)):
        ax.fill_between(sigm_e,T_conv_mean[:,k]-T_conv_err[:,k],T_conv_mean[:,k]+T_conv_err[:,k],
                        color=viridis(viridis_max*k/k_max_plot),alpha=0.3,zorder=k_max_plot-k)
        ax.plot(sigm_e,T_conv_mean[:,k],
                        color=viridis(viridis_max*k/k_max_plot),zorder=k_max_plot-k)
    

    custom_lines = [Line2D([0], [0], color=viridis(0.), lw=2),
                    Line2D([0], [0], color=viridis(viridis_max/2.), lw=2),
                    Line2D([0], [0], color=viridis(viridis_max), lw=2)]

    ax.legend(custom_lines, [('%.2f' % sigm_e[1]),
                             ('%.2f' % sigm_e[int(k_max_plot/2.)]),
                             ('%.2f' % sigm_e[k_max_plot])],
                             title=r'$R_{\rm t}$')
    
    #pcm = ax.pcolormesh(r_t_pcm,sigm_e_pcm,T_conv_mask.mean(axis=0),cmap="viridis")
    #plt.colorbar(pcm,ax=ax)
    
    ax.set_ylim([0.,1.5e4])
    
    ax.set_xlabel(r'$\sigma_{\rm ext}$')
    ax.set_ylabel(r'$T_{\rm conv}$')

if __name__ == '__main__':
    
    fig,ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH,TEXT_WIDTH*0.4))
    
    plot(ax[0],("/home/fabian/work/repositories/DrivenRandNetworkCriticality/data/"
                +"heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/T_conv_sweep_no_norm_new.npz"))
    plot(ax[1],("/home/fabian/work/repositories/DrivenRandNetworkCriticality/data/"
                +"heterogeneous_independent_gaussian_input_ESN/alt_hom_regulation/T_conv_sweep_norm_new.npz"))
    
    ax[0].get_legend().remove()
    
    fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

    ax0_title = '\\makebox['+str(ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont without renormalization}'
    ax1_title = '\\makebox['+str(ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont with renormalization}'

    ax[0].set_title(ax0_title,loc='left',usetex=True)
    ax[1].set_title(ax1_title,loc='left',usetex=True)
    
    fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)
    
    fig.savefig(os.path.join(PLOT_DIR,'T_conv_line_het_gauss_flow_local.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'T_conv_line_het_gauss_flow_local.png'),dpi=1000)
    
    plt.show()
    