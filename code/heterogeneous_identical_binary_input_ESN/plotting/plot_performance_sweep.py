#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os,glob,sys,re

from tqdm import tqdm

from src.analysis_tools import get_simfile_prop

import pandas as pd

def plot(ax):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    file_search = glob.glob(os.path.join(DATA_DIR,'heterogeneous_identical_binary_input_ESN/param_sweep_performance_*'))

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

    sweep_df = pd.DataFrame(columns=('sigm_e','sigm_t','MC_abs','specrad'))

    for k,dat_inst in enumerate(dat):

        sigm_e = dat_inst['sigm_e']
        sigm_t = dat_inst['sigm_t']

        a = dat_inst['a']
        W = dat_inst['W']

        MC_abs = dat_inst['MC'].sum(axis=2)

        n_sigm_t = sigm_t.shape[0]
        n_sigm_e = sigm_e.shape[0]

        print('Processing data...')

        for k in tqdm(range(n_sigm_e)):
            for l in range(n_sigm_t):

                specrad = np.abs(np.linalg.eigvals((a[k,l,:] * W[k,l,:,:].T).T)).max()

                sweep_df = sweep_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','MC_abs','specrad'),data=np.array([[sigm_e[k],sigm_t[l],MC_abs[k,l],specrad]])))


    sweep_df_group = sweep_df.groupby(by=['sigm_e','sigm_t'])

    sweep_df_mean = sweep_df_group.mean()
    sweep_df_mean.reset_index(inplace=True)

    sweep_df_sem = sweep_df_group.agg('sem')
    sweep_df_sem.reset_index(inplace=True)

    sweep_df_merge = pd.merge(sweep_df_mean,sweep_df_sem,on=['sigm_e','sigm_t'],suffixes=['_mean','_sem'])

    ax.pcolormesh(sigm_t,sigm_e,sweep_df_merge.pivot(index='sigm_e',columns='sigm_t',values='MC_abs_mean'))
    ax.contour(sigm_t,sigm_e,sweep_df_merge.pivot(index='sigm_e',columns='sigm_t',values='specrad_mean'),levels=[1.],linestyles=['dashed'],colors=[colors[0]],linewidths=[2.])

    sweep_df_group_sigm_e = sweep_df.groupby(by=['sigm_e'])

    sweep_df_max_sigm_t = sweep_df_group_sigm_e.agg('max')
    sweep_df_max_sigm_t.reset_index(inplace=True)
    


if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,0.7*TEXT_WIDTH))

    plot(ax)

    plt.show()
