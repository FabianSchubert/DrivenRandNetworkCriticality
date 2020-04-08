#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_act_end(ax_act):

    Data = np.load("../../data/sim_results.npz")

    n_t = Data["n_t"]
    t_ext_off = Data["t_ext_off"]

    t_ax = np.array(range(n_t-100,n_t))

    ax_act.plot(t_ax,Data["x_net_rec"][-100:,:5])
    ax_act.set_xlabel("Time Step")
    ax_act.ticklabel_format(axis='x', style='sci', useOffset=n_t-100, useMathText=True)
    ax_act.set_ylabel("Recurrent Activity")

    ax_act.set_title("E",{'fontweight' : 'bold'}, loc="left")

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

fig_act, ax_act = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

plot_act_end(ax_act)

fig_act.tight_layout()
fig_act.savefig("../../plots/act.png", dpi=300)

plt.show()
