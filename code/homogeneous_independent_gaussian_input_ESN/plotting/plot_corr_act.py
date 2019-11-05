#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')
ModCols = mpl.cm.get_cmap('viridis',512)(np.linspace(0.,.7,512))
ModCm = mpl.colors.ListedColormap(ModCols)

from stdParams import *
import os
import glob

from src.analysis_tools import get_simfile_prop

import pandas as pd

def plot(ax):

    file_search = glob.glob(os.path.join(DATA_DIR,'homogeneous_independent_gaussian_input_ESN/N_500/param_sweep_*'))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cmap = mpl.cm.get_cmap('viridis')

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

    corr_df = pd.DataFrame(columns=('sigm_e','sigm_t','cross_corr'))

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
                avg_off_diag = (np.abs(corr).sum() - np.abs(corr[range(N),range(N)]).sum())/(N**2-N)


                corr_df = corr_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','cross_corr'),data=np.array([[sigm_e[k],sigm_t[l],avg_off_diag]])))

    corr_df["sigm_e"] = ["$%s$" % x for x in corr_df["sigm_e"]]
    #mean_corr = corr_df.groupby(by=['sigm_e','sigm_t']).mean()
    #sem_corr = corr_df.groupby(by=['sigm_e','sigm_t']).agg('sem')

    #mean_corr.reset_index(inplace=True)
    #sem_corr.reset_indes(inplace=True)


    sns.lineplot(ax=ax,x='sigm_t',y='cross_corr',hue='sigm_e',data=corr_df,palette='viridis')

    ax.legend().texts[0].set_text('$\\sigma_{\\rm e}$')

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('Mean Activity Cross Correlation')



if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_independent_gaussian_input_corr_act.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_independent_gaussian_input_corr_act.png'),dpi=1000)

    plt.show()
