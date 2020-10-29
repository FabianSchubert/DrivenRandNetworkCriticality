#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\usepackage{cmbright}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *
import os

import ESN_code.plotting.plot_alt_hom_regulation_flow as plot_flow

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

plot_flow.plot(ax1,"heterogeneous_identical_binary",0,1.,"flow_data_sigm_e_0_r_t_1.npz")
plot_flow.plot(ax2,"heterogeneous_identical_binary",0,.7,"flow_data_sigm_e_005_r_t_07.npz")
plot_flow.plot(ax3,"heterogeneous_identical_binary",0,1.3,"flow_data_sigm_e_025_r_t_13.npz")
plot_flow.plot(ax4,"heterogeneous_identical_binary",0,1.,"flow_data_sigm_e_05_r_t_1.npz")

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

ax1_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont $\\sigma_{\\rm ext} = 0.0$, $R_{\\rm t} = 1$}'
ax2_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont $\\sigma_{\\rm ext} = 0.05$, $R_{\\rm t} = 0.7$}'
ax3_title = '\\makebox['+str(ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf C} \\hfill \\normalfont $\\sigma_{\\rm ext} = 0.25$, $R_{\\rm t} = 1.3$}'
ax4_title = '\\makebox['+str(ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf D} \\hfill \\normalfont $\\sigma_{\\rm ext} = 0.5$, $R_{\\rm t} = 1$}'

ax1.set_title(ax1_title,loc='left',usetex=True)
ax2.set_title(ax2_title,loc='left',usetex=True)
ax3.set_title(ax3_title,loc='left',usetex=True)
ax4.set_title(ax4_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_flow_composite.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_flow_composite.png'),dpi=300)

plt.show()
