#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from matplotlib import cm
viridis = cm.get_cmap('viridis', 12)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input_type",
help='''specify four type of input (homogeneous_identical_binary,
homogeneous_independent_gaussian, heterogeneous_identical_binary,
heterogeneous_independent_gaussian)''',
choices=['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'])

parser.add_argument("adaptation_mode",
help='''specify the mode of adaptation: local or global''',
choices=['local','global'])

def plot(ax,input_type,adaptation_mode):
    
    file_search = glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_alt_hom_regulation_performance_'
                                            +adaptation_mode
                                            +'_*'))
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
    
    sigm_e = dat[0]["sigm_e"]
    R_t = dat[0]["r_a"]
    
    n_sigm_e = sigm_e.shape[0]
    n_R_t = R_t.shape[0]
    
    n_tau = dat[0]["MC"].shape[3]
    
    MC = np.ndarray((len(dat),n_sigm_e,n_R_t,n_tau))
    
    for k in range(len(dat)):
        MC[k] = dat[k]["MC"]
    
    max_perf = R_t[np.argmax(MC,axis=2)]
    
    mean_max05 = max_perf[:,:,:5].mean(axis=(0,2))
    err_max05 = max_perf[:,:,:5].std(axis=(0,2))/5.**.5
    
    mean_max510 = max_perf[:,:,5:10].mean(axis=(0,2))
    err_max510 = max_perf[:,:,5:10].std(axis=(0,2))/5.**.5
    
    mean_max1015 = max_perf[:,:,10:15].mean(axis=(0,2))
    err_max1015 = max_perf[:,:,10:15].std(axis=(0,2))/5.**.5
    
    p05, = plt.plot(mean_max05,sigm_e,color=viridis(0),lw=1.5,label=r'$\tau = \{1,..,5\}$')
    plt.fill_betweenx(sigm_e,mean_max05-err_max05,mean_max05+err_max05,color=viridis(0),alpha=0.3)
    
    p510, = plt.plot(mean_max510,sigm_e,color=viridis(.5),lw=1.5,label=r'$\tau = \{6,..,11\}$')
    plt.fill_betweenx(sigm_e,mean_max510-err_max510,mean_max510+err_max510,color=viridis(.5),alpha=0.3)
    
    p1015, = plt.plot(mean_max1015,sigm_e,color=viridis(.9),lw=1.5,label=r'$\tau = \{12,..,16\}$')
    plt.fill_betweenx(sigm_e,mean_max1015-err_max1015,mean_max1015+err_max1015,color=viridis(.9),alpha=0.3)
    
    ax.legend(frameon=True)
    
    ax.set_xlabel(r'$R_{\rm t}$')
    ax.set_ylabel(r'$\sigma_{\rm ext}$')
    
    ax.set_ylim([sigm_e[0],sigm_e[-1]])
    ax.set_xlim([0.,1.8])
    
if __name__ == "__main__":

    args = parser.parse_args()

    fig, ax = plt.subplots(1,1,figsize=(3,2.5))

    plot(ax,args.input_type,args.adaptation_mode) 
    
    fig.tight_layout(pad=0.1)
    
    fig.savefig("/home/fabian/Nextcloud/presentation_neuronus/figures/alt_hom_regulation_max_perf_bins.pdf", facecolor="None")
    fig.savefig("/home/fabian/Nextcloud/presentation_neuronus/figures/alt_hom_regulation_max_perf_bins.png",dpi=600, facecolor="None")
    
    plt.show()
        