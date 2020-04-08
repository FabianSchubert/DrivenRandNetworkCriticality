#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot, transforms
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os,glob,sys,re

from datetime import datetime

from tqdm import tqdm

from src.analysis_tools import get_simfile_prop

from scipy.interpolate import interp2d

import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_type",
help='''specify four type of input (homogeneous_identical_binary,
homogeneous_independent_gaussian, heterogeneous_identical_binary,
heterogeneous_independent_gaussian)''',
default='homogeneous_independent_gaussian')

args = parser.parse_args()


def plot(ax,input_type):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    #file_search_preprocess,timestamp_preprocess = get_simfile_prop(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data'),return_None=True)

    #if file_search_preprocess != None:

    try:
        #sweep_df = pd.read_pickle(file_search_preprocess)
        sweep_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data.h5'), 'table')
    except:

        file_search = glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_*'))

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

        sweep_df = pd.DataFrame(columns=('sigm_e','sigm_t','MC_abs','specrad','timestamp'))

        for i,dat_inst in enumerate(dat):

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

                    sweep_df = sweep_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','MC_abs','specrad','timestamp'),data=np.array([[sigm_e[k],sigm_t[l],MC_abs[k,l],specrad,timestamp[i]]])))

        sweep_df.sigm_e = sweep_df.sigm_e.astype('float')
        sweep_df.sigm_t = sweep_df.sigm_t.astype('float')
        sweep_df.MC_abs = sweep_df.MC_abs.astype('float')
        sweep_df.specrad = sweep_df.specrad.astype('float')
        sweep_df.timestamp = sweep_df.timestamp.astype('datetime64')

        sweep_df = sweep_df.reset_index()

        #sweep_df.to_pickle(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data_'+str(datetime.now().isoformat())+'.pkl'))

        sweep_df.to_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data.h5'),'table')

    sigm_e = sweep_df.sigm_e.unique()
    sigm_t = sweep_df.sigm_t.unique()


    sweep_df_group = sweep_df.groupby(by=['sigm_e','sigm_t'])

    sweep_df_mean = sweep_df_group.mean()
    sweep_df_mean.reset_index(inplace=True)

    sweep_df_sem = sweep_df_group.agg('sem')
    sweep_df_sem.reset_index(inplace=True)

    sweep_df_merge = pd.merge(sweep_df_mean,sweep_df_sem,on=['sigm_e','sigm_t'],suffixes=['_mean','_sem'])


    sweep_df_group_sigm_e_timestamp = sweep_df.groupby(by=['sigm_e','timestamp'])

    max_MC_idx = sweep_df_group_sigm_e_timestamp.idxmax()

    max_MC_values = sweep_df.loc[max_MC_idx.MC_abs]

    max_MC_values_mean = max_MC_values.groupby(by=['sigm_e']).mean()
    max_MC_values_sem = max_MC_values.groupby(by=['sigm_e']).agg('sem')



    #sweep_df_max_sigm_t = sweep_df_group_sigm_e.agg('max')
    #sweep_df_max_sigm_t.reset_index(inplace=True)

    MC_pivot = sweep_df_merge.pivot(index='sigm_e',columns='sigm_t',values='MC_abs_mean').to_numpy()

    #max_MC_idx = np.argmax(MC_pivot,axis=1)
    #max_MC = MC_pivot[range(MC_pivot.shape[0]),max_MC_idx]

    ### Cutoff for masking is 0.2
    #pcm = ax.pcolormesh(sigm_t,sigm_e,np.ma.MaskedArray(MC_pivot,MC_pivot < 2e-1),cmap='viridis',rasterized=True,vmin=0.,vmax=9.)

    #plt.colorbar(ax=ax,mappable=pcm)

    contour = ax.contour(sigm_t,sigm_e,sweep_df_merge.pivot(index='sigm_e',columns='sigm_t',values='specrad_mean'),levels=[1.],colors=[(0.,0.,0.,0.)],linewidths=[0.])

    paths = contour.collections[0].get_paths()[0]
    vert = paths.vertices
    sigm_t_crit_manifold = vert[:,0]
    sigm_e_crit_manifold = vert[:,1]


    f_MC = interp2d(sigm_t,sigm_e,MC_pivot)

    MC_crit_manifold = []

    for k in range(sigm_t_crit_manifold.shape[0]):
        MC_crit_manifold.append(f_MC(sigm_t_crit_manifold[k],sigm_e_crit_manifold[k])[0])

    MC_crit_manifold = np.array(MC_crit_manifold)


    ax.plot(sigm_e[1:],max_MC_values_mean.MC_abs.to_numpy()[1:],lw=2.,label='Maximal performance')
    ax.fill_between(sigm_e[1:],(max_MC_values_mean.MC_abs-max_MC_values_sem.MC_abs).to_numpy()[1:],(max_MC_values_mean.MC_abs+max_MC_values_sem.MC_abs).to_numpy()[1:],alpha=.25)

    #ax.plot(sigm_e[1:],max_MC[1:],label='Maximal performance')

    ax.plot(sigm_e_crit_manifold,MC_crit_manifold,label='Performance for  $R_{\\rm a} = 1$')

    #ax.plot((1.5**.5 / sigm_e + 1.)**(-.5),sigm_e,'--',lw=2.,c=BRIGHT_GREEN)
    #sigm_e_crit = (sigm_t**2./2.**.5)*(3.**.5 + (1.-3.**.5)*sigm_t**2.)/(1.-sigm_t**2.)
    #ax.plot(sigm_t,sigm_e_crit,'--',lw=2.,c=BRIGHT_RED)

    #ax.set_xlim([sigm_t[0],sigm_t[-1]])
    ax.set_xlim([sigm_e[0],sigm_e[-2]])

    ax.set_xlabel("$\\sigma_{\\rm ext}$")
    ax.set_ylabel("$MC_{\\rm XOR}$")

    #ax.legend()


if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*.5,TEXT_WIDTH*.45))

    plot(ax,args.input_type)
    ax.set_xlabel("$\\sigma_{\\rm t}$ (target)")
    ax.set_ylabel("$\\sigma_{\\rm ext}$ (input)")

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, args.input_type + '_input_xor_perf_manifolds.pdf'))
    fig.savefig(os.path.join(PLOT_DIR, args.input_type + '_input_xor_perf_manifolds.png'),dpi=300)

    plt.show()
