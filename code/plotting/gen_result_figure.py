#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plot_act_trans import plot_act_trans
from plot_bias import plot_bias
from plot_gain import plot_gain
from plot_act_var import plot_act_var
from plot_max_eig import plot_max_eig

textwidth = 5.5532

#fig, ax = plt.subplots(3,2,figsize=(textwidth,0.7*textwidth))
fig = plt.figure(figsize=(textwidth,textwidth))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)


file = "/media/fschubert/TOSHIBA EXT/simdata/sim_results.npz"

plot_act_trans(ax1,file)
ax1.set_title("A",{'fontweight' : 'bold'}, loc="left")
plot_bias(ax2,file)
ax2.set_title("B",{'fontweight' : 'bold'}, loc="left")
plot_gain(ax3,file)
ax3.set_title("C",{'fontweight' : 'bold'}, loc="left")
plot_act_var(ax4,file)
ax4.set_title("D",{'fontweight' : 'bold'}, loc="left")
plot_max_eig(ax5,file)
ax5.set_title("E",{'fontweight' : 'bold'}, loc="left")

ax4.set_ylim([0.,2.])

fig.tight_layout()

fig.savefig("../../plots/res_comp.png",dpi=300)
fig.savefig("../../plots/res_comp.pdf")

plt.show()
