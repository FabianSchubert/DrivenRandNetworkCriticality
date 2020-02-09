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

from src.analysis_tools import get_simfile_prop

def plot(ax):

    try:
        MSE_df = pd.read_hdf(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/MSE_df.h5'), 'table')
    except:
        print("No dataframe found! Creating it...")

        file_search = glob.glob(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/N_500/param_sweep_*'))

        #####
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

        MSE_df = pd.DataFrame(columns=('sigm_e','sigm_t','MSE'))

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

                    MSE_df = MSE_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','MSE'),data=np.array([[sigm_e[k],sigm_t[l],MSE]])))

        MSE_df["sigm_e"] = ["$%s$" % x for x in MSE_df["sigm_e"]]

        MSE_df.to_hdf(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/MSE_df.h5'),'table')

    '''
    file_search = glob.glob(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/N_500/param_sweep_*'))

    #####
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

    MSE_df = pd.DataFrame(columns=('sigm_e','sigm_t','MSE'))

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

                MSE_df = MSE_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','MSE'),data=np.array([[sigm_e[k],sigm_t[l],MSE]])))

    MSE_df["sigm_e"] = ["$%s$" % x for x in MSE_df["sigm_e"]]
    '''

    sns.lineplot(ax=ax,x='sigm_t',y='MSE',hue='sigm_e',data=MSE_df,palette='viridis')

    ax.legend().texts[0].set_text('$\\sigma_{\\rm e}$')

    ax.set_xlabel('$\\sigma_{\\rm t}$')
    ax.set_ylabel('$\\left\\langle\\left[\\sigma_{\\rm X_r}-\\sigma_{\\rm y} \sigma_{\\rm w}\\right]^2\\right\\rangle_{\\rm P}$')

    #ax.legend()

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))



    plot(ax)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_rec_mem_pot_predict.pdf'))
    fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_rec_mem_pot_predict.png'),dpi=1000)

    plt.show()
