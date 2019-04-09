#!/usr/bin/env python3

from plot_std_in_std_target_sweep import plot_gain_mean_sweep, plot_gain_mean_crit_trans_sweep

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

file = "../../data/max_lyap_sweep/sim_results.npz"

Data = np.load(file)

sigm_t = Data["std_act_target_sweep_range"]
sigm_e = Data["std_in_sweep_range"]

textwidth = 5.5532
std_figsize = (textwidth * 0.7, textwidth * 0.6)


fig, ax = plt.subplots(1,1,figsize=std_figsize)

ax.set_title("${\\bf A}\\ $   Gain", loc="left")

plot_gain_mean_sweep(ax,file)
plot_gain_mean_crit_trans_sweep(ax,critval=1.,color='#FFFFFF',file_path=file)
ax.set_xticks([0.,0.5])
ax.set_yticks([0.,0.5,1.,1.5])

y = np.linspace(0.,0.8,1000)

x = np.linspace(sigm_t[0],sigm_t[-1],1000)

### Polynomial approx
ax.plot((1./8. - (y-1./8.**.5)**2)**.5,y,'--',c='#FFFF00',lw=2)

### Gaussian approx
ax.plot(x,(1./(2.*(1.-x**2.)**2.) - 0.5 - x**2.)**.5,':',c='#FFFF00',lw=2)

ax.set_ylim([sigm_e[0],sigm_e[-1]])
ax.set_xlim([sigm_t[0],sigm_t[-1]])

fig.tight_layout(pad=0.)

fig.savefig("../../plots/crit_transition_2nd_order_approx.png", dpi=1000)
fig.savefig("../../plots/crit_transition_2nd_order_approx.pdf")

plt.show()
