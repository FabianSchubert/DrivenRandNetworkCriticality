#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('matplotlibrc')

from plot_std_in_std_target_sweep import plot_max_l_sweep, plot_gain_mean_sweep, plot_gain_std_sweep, plot_3d_gain_mean_sweep

textwidth = 5.5532
std_figsize = (textwidth,textwidth*0.9)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=std_figsize)

ax = [fig.add_subplot(221)]
ax.append(fig.add_subplot(222,sharey=ax[0]))
ax.append(fig.add_subplot(223))
ax.append(fig.add_subplot(224,projection='3d'))

labels = ["A","B","C","D"]

for k in range(4):
    ax[k].set_title(labels[k],loc="left",fontweight="bold")

plot_max_l_sweep(ax[0])
plot_gain_mean_sweep(ax[1])
ax[1].set_ylabel("")
plot_gain_std_sweep(ax[2])
plot_3d_gain_mean_sweep(ax[3])
ax[3].set_xticks([0.,0.1,0.2])
ax[3].set_yticks([0.,0.5,1.])
ax[3].set_zticks([0.,0.5,1.])

#fig.tight_layout()
fig.subplots_adjust(left=0.107,right=0.9,top=0.95,bottom=0.11,hspace=0.4,wspace=0.225)
fig.savefig("../../plots/std_in_std_target_sweep_fig.png",dpi=300)

plt.show()
