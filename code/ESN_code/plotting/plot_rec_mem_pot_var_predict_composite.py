#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from matplotlib.lines import Line2D

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')


from stdParams import *
import os

import ESN_code.plotting.plot_rec_mem_pot_variance_predict_size_scaling as plot_var_pred

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax = plt.subplot(111)
#ax2 = plt.subplot(222)
#ax3 = plt.subplot(223)
#ax4 = plt.subplot(224)

print("plotting variance prediction error scaling for homogeneous_independent_gaussian...")
plot_var_pred.plot(ax,'homogeneous_independent_gaussian',col=colors[0])
print("plotting variance prediction error scaling for homogeneous_identical_binary...")
plot_var_pred.plot(ax,'homogeneous_identical_binary',col=colors[1])
print("plotting variance prediction error scaling for heterogeneous_independent_gaussian...")
plot_var_pred.plot(ax,'heterogeneous_independent_gaussian',col=colors[2])
print("plotting variance prediction error scaling for heterogeneous_identical_binary...")
plot_var_pred.plot(ax,'heterogeneous_identical_binary',col=colors[3])

ax.set_ylim([0.,.25])

custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color=colors[2], lw=2),
                Line2D([0], [0], color=colors[3], lw=2)]

ax.legend(custom_lines,['hom. gauss.',
                        'hom. bin.',
                        'het. gauss.',
                        'het. bin.'])

#ax1.set_title(ax1_title,loc='left',usetex=True)
#ax2.set_title(ax2_title,loc='left',usetex=True)
#ax3.set_title(ax3_title,loc='left',usetex=True)
#ax4.set_title(ax4_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'var_predict_composite.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'var_predict_composite.png'),dpi=300)

#fig.savefig(os.path.join(PLOT_DIR,'var_predict_composite_low_res.png'),dpi=300)

plt.show()
