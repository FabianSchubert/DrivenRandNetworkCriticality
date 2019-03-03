#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plot_act import plot_act_pop_mean, plot_act_trail_av
from plot_bias import plot_bias

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

textwidth = 5.5532

#fig, ax = plt.subplots(3,2,figsize=(textwidth,0.7*textwidth))
fig = plt.figure(figsize=(textwidth,textwidth*0.4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#ax5 = fig.add_subplot(325)

file = "../../data/sim_results_single_run.npz"

labels = ["${\\bf A}\\ $   Neural Activity",
"${\\bf B}\\ $   Bias"]

#plot_act_pop_mean(ax1,file)
plot_act_trail_av(ax1,file)
ax1.set_title(labels[0], loc="left")
plot_bias(ax2,file)
ax2.set_title(labels[1], loc="left")

fig.tight_layout(pad=0.)

fig.savefig("../../plots/res_comp_act_bias.png",dpi=1000)
fig.savefig("../../plots/res_comp_act_bias.pdf")

plt.show()
