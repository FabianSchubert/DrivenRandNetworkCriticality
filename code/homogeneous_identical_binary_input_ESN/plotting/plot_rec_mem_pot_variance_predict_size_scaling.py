#!/usr/bin/env python3

import numpy as np
import pandas as pd
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
from pathlib import Path

from src.analysis_tools import get_simfile_prop

def plot(ax):

    #####
    sizes = [100,200,300,400,500,1000]

    filelist = []

    for size in sizes:
        filelist.append(glob.glob(os.path.join(DATA_DIR,'homogeneous_identical_binary_input_ESN/N_'+str(size) + '/param_sweep_*')))
        for k,file in enumerate(filelist[-1]):
            filelist[-1][k] = Path(file).relative_to(DATA_DIR)
    #####

    MSE_df = pd.DataFrame(columns=('sigm_e','sigm_t','N','MSE'))

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
            X_r = dat_inst['X_r']
            W = dat_inst['W']

            sigm_w = W.std(axis=3)
            sigm_X_r = X_r.std(axis=2)
            sigm_y = y.std(axis=2)

            n_sigm_t = sigm_t.shape[0]
            n_sigm_e = sigm_e.shape[0]

            N = y.shape[3]

            for k in range(n_sigm_e):
                for l in range(n_sigm_t):
                    MSE = ((sigm_X_r[k,l,:]-sigm_w[k,l,:]*sigm_y[k,l,:]*N**.5)**2.).mean()

                    MSE_df = MSE_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','N','MSE'),data=np.array([[sigm_e[k],sigm_t[l],N,MSE]])))

    MSE_df = MSE_df.loc[MSE_df['sigm_e']==0.]
    MSE_df["N"] = ["$%s$" % x for x in MSE_df["N"].astype('int')]

    sns.lineplot(ax=ax,x='sigm_t',y='MSE',hue='N',data=MSE_df,palette='viridis')

    #ax.legend().texts[0].set_text('$\\sigma_{\\rm e}$')

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('$\\left\\langle\\left[\\sigma_{\\rm X_r}-\\sigma_{\\rm y} \sigma_{\\rm w}\\right]^2\\right\\rangle_{\\rm P}$')

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_identical_binary_input_rec_mem_pot_predict_size_scaling.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'homogeneous_identical_binary_input_rec_mem_pot_predict_size_scaling.png'),dpi=1000)

    plt.show()
