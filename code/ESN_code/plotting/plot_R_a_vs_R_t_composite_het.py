#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *
import os

import ESN_code.plotting.plot_R_a_vs_R_t as plot_R_a_vs_R_t

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.45))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


plot_R_a_vs_R_t.plot(ax2,'heterogeneous_independent_gaussian','local')

plot_R_a_vs_R_t.plot(ax1,'heterogeneous_identical_binary','local')

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

ax2_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous gauss}'
ax1_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous binary}'

ax1.set_title(ax1_title,loc='left',usetex=True)
ax2.set_title(ax2_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'R_a_vs_R_t_composite_het.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'R_a_vs_R_t_composite_het.png'),dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'r_a_sweep_composite_low_res.png'),dpi=300)
if not(args.hide_plot):
   plt.show()
