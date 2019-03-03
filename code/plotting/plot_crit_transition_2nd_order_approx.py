#!/usr/bin/env python3

from plot_std_in_std_target_sweep import plot_gain_mean_sweep, plot_gain_mean_crit_trans_sweep

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

file = "../../data/max_lyap_sweep/sim_results.npz"

textwidth = 5.5532
std_figsize = (textwidth * 0.7, textwidth * 0.6)


fig, ax = plt.subplots(1,1,figsize=std_figsize)

ax.set_title("${\\bf A}\\ $   Gain", loc="left")

plot_gain_mean_sweep(ax,file)
plot_gain_mean_crit_trans_sweep(ax,critval=1.,color='#FFFFFF',file_path=file)
ax.set_xticks([0.,0.5])
ax.set_yticks([0.,0.5,1.,1.5])

y = np.linspace(0.,0.8,1000)

ax.plot((1./8. - (y-1./8.**.5)**2)**.5,y,'--',c='#FFFF00',lw=2)

fig.tight_layout(pad=0.)

fig.savefig("../../plots/crit_transition_2nd_order_approx.png", dpi=1000)
fig.savefig("../../plots/crit_transition_2nd_order_approx.pdf")

plt.show()
