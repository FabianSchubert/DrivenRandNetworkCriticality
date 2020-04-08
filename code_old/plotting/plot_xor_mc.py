#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

from plot_std_in_std_target_sweep import (plot_echo_state_prop_trans as plot_esp_trans,
                                        plot_scalar_mesh,
                                        plot_max_l_crit_trans_sweep as plot_l_crit,
                                        plot_scalar_max_fixed_ext as plot_scal_max_fix_ext)


color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

textwidth = 5.5532

file = '../../data/max_lyap_sweep/xor_lyap_esp_2.npz'

figfile = '../../plots/xor_2'

formats = ['png','pdf']

fig, ax = plt.subplots(figsize=(textwidth*0.6,textwidth*0.6*.9))

plot_scalar_mesh(ax,file,'mem_cap_xor_list')

plot_l_crit(ax,color='#00AAFF',lw=3,file_path=file)

plot_esp_trans(ax,color='#FF0000',file_path=file)

plot_scal_max_fix_ext(ax,color='#FFCC00',lw=3,file_path=file,key='mem_cap_xor_list',range_y=[1,30])

fig.tight_layout(pad=0.)

for fmt in formats:
    if fmt == 'png':
        fig.savefig(figfile + '.' + fmt,dpi=1000)
    else:
        fig.savefig(figfile + '.' + fmt)


plt.show()
