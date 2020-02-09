#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

def plot(ax):

    #####
    #####
    sizes = [100,200,300,400,500,1000]

    filelist = []

    for size in sizes:
        filelist.append(glob.glob(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/N_'+str(size) + '/param_sweep_*')))
        for k,file in enumerate(filelist[-1]):
            filelist[-1][k] = Path(file).relative_to(DATA_DIR)
    #####

    corr_df = pd.DataFrame(columns=('sigm_e','sigm_t','N','corr'))

    for file_search in filelist:

        if isinstance(file_search,list):
            simfile = []
            timestamp = []
            for file_search_inst in file_search:
                simfile_inst, timestamp_inst = get_simfile_prop(os.path.join(DATA_DIR,file_search_inst))
                simfile.append(simfile_inst)
                timestamp.append(timestamp_inst)
        else:
            simfile,timestamp = get_simfile_prop(os.path.join(DATA_DIR,file_search))
            simfile = [simfile]
            timestamp = [timestamp]

        dat = []

        for simfile_inst in simfile:
            dat.append(np.load(simfile_inst))



        for dat_inst in dat:

            sigm_e = dat_inst['sigm_e']
            sigm_t = dat_inst['sigm_t']
            y = dat_inst['y']

            n_sigm_t = sigm_t.shape[0]
            n_sigm_e = sigm_e.shape[0]

            N = y.shape[3]

            for k in range(n_sigm_e):
                for l in range(n_sigm_t):
                    corr = np.corrcoef(y[k,l,:,:].T)
                    avg_off_diag = (np.abs(corr**2.).sum() - np.abs(corr[range(N),range(N)]**2.).sum())/(N**2-N)

                    corr_df = corr_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','N','corr'),data=np.array([[sigm_e[k],sigm_t[l],N,N*avg_off_diag]])))

    corr_df = corr_df.loc[corr_df['sigm_e']==0.]
    corr_df["N"] = ["$%s$" % x for x in corr_df["N"].astype('int')]

    sns.lineplot(ax=ax,x='sigm_t',y='corr',hue='N',data=corr_df,palette='viridis')

    #ax.legend().texts[0].set_text('$\\sigma_{\\rm e}$')

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('$N \\cdot \\langle$ Squared Activity Cross Correlation $\\rangle$')
    #####


if __name__ == '__main__':
    '''
    sizes = [100,200,300,400,500,1000]

    filelist = []

    for size in sizes:
        filelist.append(glob.glob(os.path.join(DATA_DIR,'homogeneous_independent_gaussian_input_ESN/N_'+str(size) + '/param_sweep_*')))
        for k,file in enumerate(filelist[-1]):
            filelist[-1][k] = Path(file).relative_to(DATA_DIR)

    '''
    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    #for file in filelist:
    plot(ax)

    ax.set_yscale('log')

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_corr_act_size_scaling_log.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_corr_act_size_scaling_log.png'),dpi=1000)

    plt.show()
