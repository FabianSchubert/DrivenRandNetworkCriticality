#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from stdParams import *
import os

import ESN_code.plotting.plot_alt_hom_regulation_check_conv_cond as check_conv_cond
import ESN_code.plotting.plot_alt_hom_regulation_r_a as r_a
import ESN_code.plotting.plot_alt_hom_regulation_specrad as specrad

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.75))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax1.set_title('A',fontdict={'fontweight':'bold'},loc='left')
ax2.set_title('B',fontdict={'fontweight':'bold'},loc='left')
ax3.set_title('C',fontdict={'fontweight':'bold'},loc='left')
ax4.set_title('D',fontdict={'fontweight':'bold'},loc='left')

r_a.plot(ax1,'heterogeneous_identical_binary','local',col=colors[0])
ax1.set_ylim([0.,5.])

r_a.plot(ax2,'heterogeneous_independent_gaussian','local',col=colors[1])

r_a.plot(ax3,'heterogeneous_identical_binary','global',col=colors[2])

specrad.plot(ax4,'heterogeneous_identical_binary','local','A',col=colors[0])
specrad.plot(ax4,'heterogeneous_independent_gaussian','local','B',col=colors[1])
specrad.plot(ax4,'heterogeneous_identical_binary','global','C',col=colors[2])

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite.png'),dpi=1000)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite_low_res.png'),dpi=300)

plt.show()
