#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os

import heterogeneous_identical_binary_input_ESN.plotting.plot_corr_act as plot_corr_act
import heterogeneous_identical_binary_input_ESN.plotting.plot_gains_sweep as plot_gains_sweep
import heterogeneous_identical_binary_input_ESN.plotting.plot_rec_mem_pot_variance_predict as plot_rec_mem_pot_variance_predict

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)

print("plotting corr_act.py...")
plot_corr_act.plot(ax1)
print("plotting plot_rec_mem_pot_variance_predict.py...")
plot_rec_mem_pot_variance_predict.plot(ax2)
print("plotting gains_sweep.py...")
plot_gains_sweep.plot(ax3)

ax1.set_title('A',fontdict={'fontweight':'bold'},loc='left')
ax2.set_title('B',fontdict={'fontweight':'bold'},loc='left')
ax3.set_title('C',fontdict={'fontweight':'bold'},loc='left')

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_compos.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'heterogeneous_identical_binary_input_compos.png'),dpi=1000)

plt.show()
