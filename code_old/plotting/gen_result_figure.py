#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plot_act_trans import plot_act_trans
from plot_bias import plot_bias
from plot_gain import plot_gain
from plot_act_var import plot_act_var
from plot_max_eig import plot_max_eig

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

textwidth = 5.5532

#fig, ax = plt.subplots(3,2,figsize=(textwidth,0.7*textwidth))
fig = plt.figure(figsize=(textwidth,textwidth*0.7))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
#ax5 = fig.add_subplot(325)

file = "../../data/sim_results_single_run.npz"

labels = ["${\\bf A}\\ $   Transient Activity",
"${\\bf B}\\ $   Gain",
"${\\bf C}\\ $   Activity Variance",
"${\\bf D}\\ $   Log of Max. Lyap. Exp."]

plot_act_trans(ax1,file)
ax1.set_title(labels[0], loc="left")
plot_gain(ax2,file)
ax2.set_title(labels[1], loc="left")
plot_act_var(ax3,file)
ax3.set_title(labels[2], loc="left")
plot_max_eig(ax4,file)
ax4.set_title(labels[3], loc="left")
#plot_bias(ax5,file)
#ax5.set_title("E",{'fontweight' : 'bold'}, loc="left")

ax3.set_ylim([0.,2.])

fig.tight_layout(pad=0.8)

fig.savefig("../../plots/res_comp.png",dpi=2000)
fig.savefig("../../plots/res_comp.pdf")

plt.show()
