#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

Data = np.load("../../data/sim_results.npz")

n_t = Data["n_t"]
t_ext_off = Data["t_ext_off"]

t_ax = np.array(range(n_t-100,n_t))

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

#=====================================
fig_act_trans, ax_act_trans = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

t_pm = 100

t_ax_trans = np.array(range(t_ext_off - t_pm, t_ext_off + t_pm))

ax_act_trans.plot(t_ax_trans,Data["x_net_rec"][t_ext_off - t_pm:t_ext_off + t_pm,:5])
ax_act_trans.set_xlabel("Time Step")
ax_act_trans.ticklabel_format(axis='x', style='sci', useOffset=t_ext_off, useMathText=True)
ax_act_trans.set_ylabel("Recurrent Activity")

ax_act_trans.set_title("A",{'fontweight' : 'bold'}, loc="left")

fig_act_trans.tight_layout()
fig_act_trans.savefig("../../plots/act_trans.png", dpi=300)

plt.show()
