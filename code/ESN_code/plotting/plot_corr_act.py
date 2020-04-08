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

def plot(ax,input_type):

    #check if there is already a saved dataframe...
    try:
        corr_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/corr_df.h5'), 'table')
    except:
        print("No dataframe found! Creating it...")

        file_search = glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/gains_sweep/param_sweep_*'))

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

        for k,dat_inst in enumerate(dat):

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

        corr_df.to_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/corr_df.h5'),'table')


    sns.lineplot(ax=ax,x='sigm_t',y='cross_corr',hue='sigm_e',data=corr_df,palette='viridis')

    if input_type == 'homogeneous_independent_gaussian' or input_type == 'homogeneous_identical_binary':
        mean_corr_df = corr_df.groupby(['sigm_e','sigm_t']).agg('mean')
        mean_corr_df.reset_index(inplace=True)

        if input_type == 'homogeneous_independent_gaussian':

            corr_thresh = 0.0255

            min_over_thresh_sigm_t_0p5 = mean_corr_df.loc[mean_corr_df['sigm_e'] == '$0.5$'].loc[mean_corr_df['cross_corr'] >= corr_thresh]['sigm_t'].iloc[0]
            min_over_thresh_sigm_t_1p5 = mean_corr_df.loc[mean_corr_df['sigm_e'] == '$1.5$'].loc[mean_corr_df['cross_corr'] >= corr_thresh]['sigm_t'].iloc[0]



        else:

            corr_thresh = 1.-1e-3

            min_over_thresh_sigm_t_0p5 = mean_corr_df.loc[mean_corr_df['sigm_e'] == '$0.5$'].loc[mean_corr_df['cross_corr'] <= corr_thresh]['sigm_t'].iloc[0]
            min_over_thresh_sigm_t_1p5 = mean_corr_df.loc[mean_corr_df['sigm_e'] == '$1.5$'].loc[mean_corr_df['cross_corr'] <= corr_thresh]['sigm_t'].iloc[0]

        cmap = mpl.cm.get_cmap('viridis')

        ax.plot([min_over_thresh_sigm_t_0p5],[corr_thresh],'.',markerfacecolor=(0.,0.,0.,0.),markeredgecolor=cmap(0.5),markersize=10,markeredgewidth=1.5)
        ax.plot([min_over_thresh_sigm_t_1p5],[corr_thresh],'.',markerfacecolor=(0.,0.,0.,0.),markeredgecolor=cmap(0.75),markersize=10,markeredgewidth=1.5)


    ax.legend().texts[0].set_text('$\\sigma_{\\rm ext}$')
    ax.set_xlabel('$\\sigma_{\\rm t}$')
    #ax.set_ylabel('$\\langle \\, \\left|| C\\left(y_i,y_j\\right) \\right|| \\, \\rangle_{\\rm P}$')
    ax.set_ylabel('$\\overline{C}$')
    
    ax.set_yscale("log")


if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax,'heterogeneous_identical_binary')

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, 'heterogeneous_identical_binary' + '_input_corr_act.pdf'))
    fig.savefig(os.path.join(PLOT_DIR, 'heterogeneous_identical_binary' + '_input_corr_act.png'),dpi=1000)

    plt.show()
