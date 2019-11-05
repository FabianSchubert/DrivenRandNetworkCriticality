#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os

import homogeneous_independent_gaussian_input_ESN.plotting.plot_corr_act as plot_corr_act
import homogeneous_independent_gaussian_input_ESN.plotting.plot_gains_sweep as plot_gains_sweep
import homogeneous_independent_gaussian_input_ESN.plotting.plot_rec_mem_pot_variance_predict as plot_rec_mem_pot_variance_predict

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)

plot_corr_act.plot(ax1)
plot_rec_mem_pot_variance_predict.plot(ax2)
plot_gains_sweep.plot(ax3)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'homogeneous_independent_gaussian_input_compos.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'homogeneous_independent_gaussian_input_compos.png'),dpi=1000)

plt.show()
