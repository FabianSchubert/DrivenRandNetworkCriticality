#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
from plot_std_in_std_target_sweep import (plot_max_l_sweep,
                                          plot_gain_mean_sweep,
                                          plot_gain_std_sweep,
                                          plot_3d_gain_mean_sweep,
                                          plot_2d_gain_mean_sweep,
                                          plot_3d_gain_mean_sweep_full_tanh_pred,
                                          plot_2d_gain_mean_sweep_full_tanh_pred,
                                          plot_rmsqe_hom,
                                          plot_mem_cap,
                                          plot_max_l_crit_trans_sweep,
                                          plot_gain_mean_crit_trans_sweep,
                                          plot_echo_state_prop_trans,
                                          plot_mem_cap_max_fixed_ext)

from plot_gain_std_conv_sweep import plot_gain_std_conv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import argparse

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')


color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

file = "../../data/max_lyap_sweep/sim_results.npz"

file_std_conv = "../../data/gain_conv_sweep.npz"

figure_file = "../../plots/std_in_std_target_sweep"
figure_file_cut = "../../plots/std_in_std_target_sweep_fig_cut"

output_formats = ['pdf','png']

parser = argparse.ArgumentParser(description='''Plot the results of the parameter sweep simulation.''')

parser.add_argument('--file', help='specify file location (full path).')
parser.add_argument('--output', help='specify output file (full path without data type ending)')
parser.add_argument('--output_cut', help='specify output file for sweep cut (full path without data type ending)')

args = parser.parse_args()

if args.file!=None:
    file = args.file
if args.output!=None:
    figure_file = args.output
if args.output_cut!=None:
    figure_file_cut = args.output_cut

#file = "../../data/max_lyap_sweep/sim_results.npz"

Data = np.load(file)
std_e = Data["std_in_sweep_range"]
std_act_t = Data["std_act_target_sweep_range"]


textwidth = 5.5532
std_figsize = (textwidth, textwidth * .9)
cut_figsize = (textwidth, textwidth * .5)

fig = plt.figure(figsize=std_figsize)

ax = [fig.add_subplot(221)]
ax.append(fig.add_subplot(222, sharey=ax[0]))
ax.append(fig.add_subplot(223))
ax.append(fig.add_subplot(224))

fig_cut = plt.figure(figsize=cut_figsize)
ax_cut = [fig_cut.add_subplot(121)]
ax_cut.append(fig_cut.add_subplot(122))
#ax.append(fig.add_subplot(326))
#ax.append(fig.add_subplot(326))

labels = ["${\\bf A}\\ $   Spectral Radius",
"${\\bf B}\\ $   Gain",
"${\\bf C}\\ $   Gain Variance",
"${\\bf D}\\ $   Memory Capacity"]

labels_cut = ["${\\bf A}\\ $   Gain Linear Prediction",
"${\\bf B}\\ $   Gain Full Prediction"
]

for k in range(4):
    ax[k].set_title(labels[k], loc="left")#, fontweight="bold")

for k in range(2):
    ax_cut[k].set_title(labels_cut[k], loc="left")

plot_max_l_sweep(ax[0],file_path=file)
plot_max_l_crit_trans_sweep(ax[0],color='#00FFFF',file_path=file)
ax[0].set_xticks([0.,0.5])
ax[0].set_yticks([0.,0.5,1.,1.5])


#ax[0].set_xlabel("")

plot_gain_mean_sweep(ax[1],file_path=file)
plot_max_l_crit_trans_sweep(ax[1],color='#00FFFF',file_path=file)
plot_gain_mean_crit_trans_sweep(ax[1],color='#FFFFFF',file_path=file)
#ax[1].set_ylabel("")
#ax[1].set_xlabel("")

#plot_gain_std_sweep(ax[2],file_path=file)
plot_gain_std_conv(ax[2],file=file_std_conv)



plot_mem_cap(ax[3],file_path=file)
plot_echo_state_prop_trans(ax[3],color='#FF0000',file_path=file)
plot_max_l_crit_trans_sweep(ax[3],color='#00FFFF',file_path=file)
plot_mem_cap_max_fixed_ext(ax[3],color='#FFCC00',file_path=file)

ax[3].set_ylim([std_e[1],std_e[-1]])
ax[3].set_xticks([0.,0.5])
ax[3].set_yticks([0.,0.5,1.,1.5])

#ax[3].set_ylabel("")
#ax[3].set_xlabel("")
#plot_rmsqe_hom(ax[5])

fig.tight_layout(pad=0.)
#fig.subplots_adjust(left=0.15, right=0.956, top=0.95,
#                   bottom=0.11, hspace=0.6, wspace=0.405)
for format in output_formats:
    if format=="png":
        fig.savefig(figure_file+"."+format, dpi=1000)
    else:
        fig.savefig(figure_file+"."+format)

#fig.savefig("../../plots/std_in_std_target_sweep_fig.png", dpi=1000)
#fig.savefig("../../plots/std_in_std_target_sweep_fig.pdf")



cmap = plt.cm.viridis
custom_lines_labels = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.4), lw=4),
                Line2D([0], [0], color=cmap(.8), lw=4)]

plot_2d_gain_mean_sweep(ax_cut[0],9,colorsim=cmap(0.),colorpred=cmap(0.),file_path=file)
plot_2d_gain_mean_sweep(ax_cut[0],19,colorsim=cmap(.4),colorpred=cmap(.4),file_path=file)
plot_2d_gain_mean_sweep(ax_cut[0],29,colorsim=cmap(.8),colorpred=cmap(.8),file_path=file)

ax_cut[0].legend(custom_lines_labels, ['$\\sigma_{\\rm ext} =$ 0.5', '$\\sigma_{\\rm ext} =$ 1.0', '$\\sigma_{\\rm ext} =$ 1.5'])

#plot_3d_gain_mean_sweep(ax[3],file_path=file)
#ax[3].set_xticks([std_act_t[0], std_act_t[[0,-1]].mean() , std_act_t[-1]])
#ax[3].set_yticks([std_e[0], std_e[[0,-1]].mean() , std_e[-1]])
#ax[3].set_zticks([0., 0.5*Data["gain_list"].mean(axis=2).max(), Data["gain_list"].mean(axis=2).max()])


plot_2d_gain_mean_sweep_full_tanh_pred(ax_cut[1],9,colorsim=cmap(0.),colorpred=cmap(0.),file_path=file)
plot_2d_gain_mean_sweep_full_tanh_pred(ax_cut[1],19,colorsim=cmap(.4),colorpred=cmap(.4),file_path=file)
plot_2d_gain_mean_sweep_full_tanh_pred(ax_cut[1],29,colorsim=cmap(.8),colorpred=cmap(.8),file_path=file)



SIGM_T, SIGM_E = np.meshgrid(std_act_t,std_e)

a_exp_pred = ((1.-(1.-SIGM_T**2.)**2.)/(2.*(1.-SIGM_T**2.)**2.*(SIGM_E**2.+SIGM_T**2.)))**.5

ax_cut[1].plot(std_act_t,a_exp_pred[9,:],'--',color=cmap(0.))
ax_cut[1].plot(std_act_t,a_exp_pred[19,:],'--',color=cmap(0.4))
ax_cut[1].plot(std_act_t,a_exp_pred[29,:],'--',color=cmap(0.8))

ax_cut[1].legend(custom_lines_labels, ['$\\sigma_{\\rm ext} = 0.5$', '$\\sigma_{\\rm ext} = 1.0$', '$\\sigma_{\\rm ext} = 1.5$'])
#ax[5].set_ylabel("")
#plot_3d_gain_mean_sweep_full_tanh_pred(ax[5],file_path=file)
#ax[5].set_xticks([std_act_t[0], std_act_t[[0,-1]].mean() , std_act_t[-1]])
#ax[5].set_yticks([std_e[0], std_e[[0,-1]].mean() , std_e[-1]])
#ax[5].set_zticks([0., 0.5*Data["gain_list"].mean(axis=2).max(), Data["gain_list"].mean(axis=2).max()])

fig_cut.tight_layout(pad=0.)
#fig_cut.subplots_adjust(left=0.15, right=0.956, top=0.95,
#                   bottom=0.11, hspace=0.6, wspace=0.405)

#fig.subplots_adjust(left=0.15, right=0.956, top=0.95,
#                   bottom=0.11, hspace=0.6, wspace=0.405)

for format in output_formats:
    if format=="png":
        fig_cut.savefig(figure_file_cut+"."+format, dpi=1000)
    else:
        fig_cut.savefig(figure_file_cut+"."+format)






plt.show()
