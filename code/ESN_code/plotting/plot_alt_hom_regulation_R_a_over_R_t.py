#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot, transforms
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection

#import matplotlib.ticker as plticker

import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os,glob,sys,re

from datetime import datetime

from tqdm import tqdm

from src.analysis_tools import get_simfile_prop

import pandas as pd

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_type",
help='''specify four type of input (homogeneous_identical_binary,
homogeneous_independent_gaussian, heterogeneous_identical_binary,
heterogeneous_independent_gaussian)''',
default='homogeneous_independent_gaussian')

parser.add_argument("--adapt_mode",
help='''specify the adaptation mode (local or global).''',
default='local')

def r_t_cat(row):
    if((row["r_a"] >= 0.45) & (row["r_a"] <= 0.55)):
        return 0.5
    elif((row["r_a"] >= 0.95) & (row["r_a"] <= 1.05)):
        return 1.0
    elif((row["r_a"] >= 1.45) & (row["r_a"] <= 1.55)):
        return 1.5
    else:
        return np.nan


def plot(ax,input_type,adapt_mode):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    #file_search_preprocess,timestamp_preprocess = get_simfile_prop(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data'),return_None=True)

    #if file_search_preprocess != None:

    try:
        #sweep_df = pd.read_pickle(file_search_preprocess)
        sweep_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_alt_hom_regulation_processed_data_' +
        adapt_mode + '.h5'), 'table')
    except:
        file_search = glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_alt_hom_regulation_performance_' +
        adapt_mode + '_*'))

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

        sweep_df = pd.DataFrame(columns=('sigm_e','r_a','MC_abs','specrad','timestamp'))

        for i,dat_inst in enumerate(dat):

            sigm_e = dat_inst['sigm_e']
            r_a = dat_inst['r_a']

            a = dat_inst['a']
            W = dat_inst['W']

            MC_abs = dat_inst['MC'].sum(axis=2)

            n_r_a = r_a.shape[0]
            n_sigm_e = sigm_e.shape[0]

            print('Processing data...')

            for k in tqdm(range(n_sigm_e)):
                for l in range(n_r_a):

                    specrad = np.abs(np.linalg.eigvals((a[k,l,:] * W[k,l,:,:].T).T)).max()

                    sweep_df = sweep_df.append(pd.DataFrame(columns=('sigm_e','r_a','MC_abs','specrad','timestamp'),data=np.array([[sigm_e[k],r_a[l],MC_abs[k,l],specrad,timestamp[i]]])))

        sweep_df.sigm_e = sweep_df.sigm_e.astype('float')
        sweep_df.r_a = sweep_df.r_a.astype('float')
        sweep_df.MC_abs = sweep_df.MC_abs.astype('float')
        sweep_df.specrad = sweep_df.specrad.astype('float')
        sweep_df.timestamp = sweep_df.timestamp.astype('datetime64')

        sweep_df = sweep_df.reset_index()

        #sweep_df.to_pickle(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_processed_data_'+str(datetime.now().isoformat())+'.pkl'))

        sweep_df.to_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_alt_hom_regulation_processed_data_' +
        adapt_mode + '.h5'),'table')

    '''
    sigm_e = sweep_df.sigm_e.unique()
    R_t = sweep_df.R_t.unique()


    sweep_df_group = sweep_df.groupby(by=['sigm_e','R_t'])

    sweep_df_mean = sweep_df_group.mean()
    sweep_df_mean.reset_index(inplace=True)

    sweep_df_sem = sweep_df_group.agg('sem')
    sweep_df_sem.reset_index(inplace=True)

    sweep_df_merge = pd.merge(sweep_df_mean,sweep_df_sem,on=['sigm_e','R_t'],suffixes=['_mean','_sem'])


    sweep_df_group_sigm_e_timestamp = sweep_df.groupby(by=['sigm_e','timestamp'])


    max_MC_idx = sweep_df_group_sigm_e_timestamp.idxmax()

    max_MC_values = sweep_df.loc[max_MC_idx.MC_abs]

    max_MC_values_mean = max_MC_values.groupby(by=['sigm_e']).mean()
    max_MC_values_sem = max_MC_values.groupby(by=['sigm_e']).agg('sem')

    #sweep_df_max_R_t = sweep_df_group_sigm_e.agg('max')
    #sweep_df_max_R_t.reset_index(inplace=True)

    MC_pivot = sweep_df_merge.pivot(index='sigm_e',columns='R_t',values='MC_abs_mean')

    ### Cutoff for masking is 0.2
    pcm = ax.pcolormesh(R_t,sigm_e,np.ma.MaskedArray(MC_pivot,MC_pivot < 2e-1),cmap='viridis',rasterized=True,vmin=0.,vmax=9.)

    plt.colorbar(ax=ax,mappable=pcm)

    ax.contour(R_t,sigm_e,sweep_df_merge.pivot(index='sigm_e',columns='R_t',values='specrad_mean'),levels=[1.],linestyles=['dashed'],colors=['w'],linewidths=[2.])
    '''
    sweep_df['r_t_cat'] = sweep_df.apply (lambda row: r_t_cat(row), axis=1)

    sweep_df['specrad/r_a'] = sweep_df['specrad'] / sweep_df['r_a']



    sns.lineplot(ax=ax,data=sweep_df,x="sigm_e",y="specrad/r_a",hue="r_t_cat",legend="full",palette="viridis",hue_norm=(0.5,1.8))

    h,l = ax.get_legend_handles_labels()
    #n_h = len(h)
    ax.legend(h[:],l[:],title="$R_{\\rm t}$")

    #loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    #ax.yaxis.set_major_locator(loc)

    ax.plot([0.,1,],[1.,1.],'--',c='b')

    ax.set_xlabel("$\\sigma_{\\rm ext}$")
    ax.set_ylabel("$R_{\\rm a}/R_{\\rm t}$")

    #import pdb; pdb.set_trace()

    #ax.plot(max_MC_values_mean.R_t.to_numpy()[1:],sigm_e[1:],lw=2.,c=BRIGHT_YELLOW)
    #ax.fill_betweenx(sigm_e[1:],(max_MC_values_mean.R_t-max_MC_values_sem.R_t).to_numpy()[1:],(max_MC_values_mean.R_t+max_MC_values_sem.R_t).to_numpy()[1:],color=BRIGHT_YELLOW,alpha=.25)

    #ax.plot((1.5**.5 / sigm_e + 1.)**(-.5),sigm_e,'--',lw=2.,c=BRIGHT_GREEN)
    #sigm_e_crit = (R_t**2./2.**.5)*(3.**.5 + (1.-3.**.5)*sigm_t**2.)/(1.-sigm_t**2.)
    #ax.plot(sigm_t,sigm_e_crit,'--',lw=2.,c=BRIGHT_RED)

    #ax.set_xlim([R_t[0],R_t[-1]])
    #ax.set_ylim([sigm_e[0],sigm_e[-1]])
    #ax.set_ylim([0.,1.])

    #ax.set_xlabel("$R_{\\rm t}$")
    #ax.set_ylabel("$\\sigma_{\\rm ext}$")

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*.5,TEXT_WIDTH*.45))

    args = parser.parse_args()

    plot(ax,args.input_type,args.adapt_mode)


    fig.tight_layout(pad=0.1)

    #fig.savefig(os.path.join(PLOT_DIR, 'R_t_'+args.adapt_mode + '_' + args.input_type + '_input_xor_perf.pdf'))
    #fig.savefig(os.path.join(PLOT_DIR, 'R_t_'+args.adapt_mode + '_' + args.input_type + '_input_xor_perf.png'),dpi=300)

    plt.show()
