#!/usr/bin/env python3

from plot_std_in_std_target_sweep import plot_gain_mean_sweep

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('matplotlibrc')


textwidth = 5.5532
std_figsize = (textwidth * 0.6, textwidth * 0.5)


fig, ax = plt.subplots(1,1,figsize=std_figsize)

plot_gain_mean_sweep(ax)

y = np.linspace(0.,0.5,1000)

plt.plot((1./8. - (y-1./8.**.5)**2)**.5,y,'--',c='#FFFF00',lw=2)

fig.tight_layout()

fig.savefig("../../plots/crit_transition_2nd_order_approx.png", dpi=300)

plt.show()
