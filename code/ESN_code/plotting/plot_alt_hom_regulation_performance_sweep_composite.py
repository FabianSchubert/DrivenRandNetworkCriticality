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

import ESN_code.plotting.plot_alt_hom_regulation_performance_sweep as plot_performance_sweep

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

print("plotting performance sweep homogeneous_independent_gaussian...")
plot_performance_sweep.plot(ax1,'homogeneous_independent_gaussian','local')
print("plotting performance sweep homogeneous_identical_binary...")
plot_performance_sweep.plot(ax2,'homogeneous_identical_binary','local')
print("plotting performance sweep heterogeneous_independent_gaussian...")
plot_performance_sweep.plot(ax3,'heterogeneous_independent_gaussian','local')
print("plotting performance sweep heterogeneous_identical_binary...")
plot_performance_sweep.plot(ax4,'heterogeneous_identical_binary','local')

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

ax1_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont hom. gauss.}'
ax2_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont hom. bin.}'
ax3_title = '\\makebox['+str(ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf C} \\hfill \\normalfont het. gauss.}'
ax4_title = '\\makebox['+str(ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf D} \\hfill \\normalfont het. bin.}'

ax1.set_title(ax1_title,loc='left',usetex=True)
ax2.set_title(ax2_title,loc='left',usetex=True)
ax3.set_title(ax3_title,loc='left',usetex=True)
ax4.set_title(ax4_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_performance_sweep_composite.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_performance_sweep_composite.png'),dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'r_a_sweep_composite_low_res.png'),dpi=300)

plt.show()
