#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os

import ESN_code.plotting.plot_gains_sweep as plot_gains_sweep

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

print("plotting gain sweep homogeneous_independent_gaussian...")
plot_gains_sweep.plot(ax1,'homogeneous_independent_gaussian')
print("plotting gain sweep homogeneous_identical_binary...")
plot_gains_sweep.plot(ax2,'homogeneous_identical_binary')
print("plotting gain sweep heterogeneous_independent_gaussian...")
plot_gains_sweep.plot(ax3,'heterogeneous_independent_gaussian')
print("plotting gain sweep heterogeneous_identical_binary...")
plot_gains_sweep.plot(ax4,'heterogeneous_identical_binary')

ax1.set_title('A',fontdict={'fontweight':'bold'},loc='left')
ax2.set_title('B',fontdict={'fontweight':'bold'},loc='left')
ax3.set_title('C',fontdict={'fontweight':'bold'},loc='left')
ax4.set_title('D',fontdict={'fontweight':'bold'},loc='left')

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'gain_sweep_compos.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'gain_sweep_compos.png'),dpi=1000)

fig.savefig(os.path.join(PLOT_DIR,'gain_sweep_compos_low_res.png'),dpi=300)

plt.show()
