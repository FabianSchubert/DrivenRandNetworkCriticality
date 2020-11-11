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

import ESN_code.plotting.plot_alt_hom_regulation_delta_R_a_R_t as plot_delta_R_a_R_t

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.9))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)


plot_delta_R_a_R_t.plot(ax2,'heterogeneous_independent_gaussian','local')

plot_delta_R_a_R_t.plot(ax1,'heterogeneous_identical_binary','local')

plot_delta_R_a_R_t.plot(ax4,'homogeneous_independent_gaussian','local')

plot_delta_R_a_R_t.plot(ax3,'homogeneous_identical_binary','local')

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

ax2_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous gauss}'
ax1_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous binary}'

ax4_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf D} \\hfill \\normalfont homogeneous gauss}'
ax3_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf C} \\hfill \\normalfont homogeneous binary}'


ax1.set_title(ax1_title,loc='left',usetex=True)
ax2.set_title(ax2_title,loc='left',usetex=True)
ax3.set_title(ax3_title,loc='left',usetex=True)
ax4.set_title(ax4_title,loc='left',usetex=True)

ax1.set_ylim([-1.,1.])
ax2.set_ylim([-1.,1.])
ax3.set_ylim([-1.,1.])
ax4.set_ylim([-1.,1.])

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_delta_R_a_R_t_composite_het_hom.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_delta_R_a_R_t_composite_het_hom.png'),dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'r_a_sweep_composite_low_res.png'),dpi=300)
if not(args.hide_plot):
   plt.show()
