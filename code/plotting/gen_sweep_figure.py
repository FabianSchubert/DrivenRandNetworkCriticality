#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
from plot_std_in_std_target_sweep import (plot_max_l_sweep,
                                          plot_gain_mean_sweep,
                                          plot_gain_std_sweep,
                                          plot_3d_gain_mean_sweep,
                                          plot_3d_gain_mean_sweep_full_tanh_pred,
                                          plot_rmsqe_hom,
                                          plot_mem_cap,
                                          plot_max_l_crit_trans_sweep,
                                          plot_gain_mean_crit_trans_sweep,
                                          plot_echo_state_prop_trans,
                                          plot_mem_cap_max_fixed_ext)
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('matplotlibrc')

#file = "/media/fschubert/TOSHIBA EXT/simdata/sim_results.npz"

file = "../../data/max_lyap_sweep/sim_results.npz"

Data = np.load(file)
std_e = Data["std_in_sweep_range"]
std_act_t = Data["std_act_target_sweep_range"]


textwidth = 5.5532
std_figsize = (textwidth, textwidth * 1.1)


fig = plt.figure(figsize=std_figsize)

ax = [fig.add_subplot(321)]
ax.append(fig.add_subplot(322, sharey=ax[0]))
ax.append(fig.add_subplot(323))
ax.append(fig.add_subplot(324, projection='3d'))
ax.append(fig.add_subplot(325))
ax.append(fig.add_subplot(326, projection='3d'))
#ax.append(fig.add_subplot(326))

labels = ["A", "B", "C", "D", "E", "F"]

for k in range(6):
    ax[k].set_title(labels[k], loc="left", fontweight="bold")

plot_max_l_sweep(ax[0],file_path=file)
plot_max_l_crit_trans_sweep(ax[0],file_path=file)

plot_gain_mean_sweep(ax[1],file_path=file)
plot_gain_mean_crit_trans_sweep(ax[1],file_path=file)
ax[1].set_ylabel("")
plot_gain_std_sweep(ax[2],file_path=file)

plot_3d_gain_mean_sweep(ax[3],file_path=file)
ax[3].set_xticks([std_act_t[0], std_act_t[[0,-1]].mean() , std_act_t[-1]])
ax[3].set_yticks([std_e[0], std_e[[0,-1]].mean() , std_e[-1]])
ax[3].set_zticks([0., 0.5*Data["gain_list"].mean(axis=2).max(), Data["gain_list"].mean(axis=2).max()])


plot_3d_gain_mean_sweep_full_tanh_pred(ax[5],file_path=file)
ax[5].set_xticks([std_act_t[0], std_act_t[[0,-1]].mean() , std_act_t[-1]])
ax[5].set_yticks([std_e[0], std_e[[0,-1]].mean() , std_e[-1]])
ax[5].set_zticks([0., 0.5*Data["gain_list"].mean(axis=2).max(), Data["gain_list"].mean(axis=2).max()])

plot_mem_cap(ax[4],file_path=file)
plot_echo_state_prop_trans(ax[4],color='#FF0000',file_path=file)
plot_max_l_crit_trans_sweep(ax[4],color='#00FF00',file_path=file)
plot_mem_cap_max_fixed_ext(ax[4],color='#FFCC00',file_path=file)


ax[4].set_ylim([std_e[1],std_e[-1]])
#plot_rmsqe_hom(ax[5])

fig.tight_layout()
fig.subplots_adjust(left=0.107, right=0.9, top=0.95,
                   bottom=0.11, hspace=0.4, wspace=0.225)
fig.savefig("../../plots/std_in_std_target_sweep_fig.png", dpi=300)
fig.savefig("../../plots/std_in_std_target_sweep_fig.pdf")

plt.show()
