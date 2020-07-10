#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os
import glob

fig, ax = plt.subplots(1,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.4))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

modes_full = [["heterogeneous","binary"],
        ["homogeneous","binary"],
        ["heterogeneous","gaussian"],
        ["homogeneous","gaussian"]]

colors_new_order = [colors[3],colors[1],colors[2],colors[0]]


cmap = mpl.cm.get_cmap('viridis')

for ax_i,k in enumerate([0,2]):
    dat = np.load("corr_R_"+modes_new_order[k][0]+"_" + modes_new_order[k][1] + ".npz",allow_pickle=True)
    
    sigm_ext_sweep = dat["sigm_ext_sweep"]
    n_sigm_ext_sweep = sigm_ext_sweep.shape[0]
    
    R_sweep = dat["R_sweep"]
    #n_R_sweep = R_sweep.shape[0]
    
    #N_sweep = dat["N_sweep"]
    #n_N_sweep = N_sweep.shape[0]
    
    Corr_av_samples = dat["Corr_av_samples"]
    
    
    
    Corr_av_mean = Corr_av_samples.mean(axis=2)
    Corr_av_err = Corr_av_samples.std(axis=2)/Corr_av_samples.shape[2]**.5
    
    for k in range(n_sigm_ext_sweep):  
           
        c = cmap(0.2+0.5*sigm_ext_sweep[k]/sigm_ext_sweep[-1])
    
        ax[ax_i].fill_between(R_sweep,Corr_av_mean[k]-Corr_av_err[k],Corr_av_mean[k]+Corr_av_err[k],alpha=.5,color=c)
        ax[ax_i].plot(R_sweep,Corr_av_mean[k],color=c,label="$\\sigma_{\\rm ext} = " + str(sigm_ext_sweep[k]) + "$")

    ax[ax_i].legend()

    ax[ax_i].set_yscale("log")
    
    ax[ax_i].set_ylabel('$\\overline{C}$')
    ax[ax_i].set_xlabel('$R_{\\rm a}$')
    
#ax.set_xscale("log")

#ax.set_xlabel('$N$')
#ax.set_ylabel('$\\left||\\sigma^2_{\\rm bare}-\sigma^2_{\\rm w} \\sigma^2_{\\rm y} \\right||$')

#ax.set_ylim([0.001,0.2])

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

ax1_title = '\\makebox['+str(ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous binary}'
ax2_title = '\\makebox['+str(ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous gauss}'

ax[0].set_title(ax1_title,loc='left',usetex=True)
ax[1].set_title(ax2_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.tight_layout(pad=0.1,h_pad=.5,w_pad=0.5)

fig.savefig("corr_R_het.pdf")
fig.savefig("corr_R_het.png",dpi=1000)

plt.show()