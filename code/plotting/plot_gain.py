#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_gain(ax_gain):

    Data = np.load("../../data/sim_results.npz")

    n_t = Data["n_t"]
    t_ext_off = Data["t_ext_off"]

    t_ax = np.array(range(n_t-100,n_t))

    ax_gain.plot(Data["gain_rec"][:,::10])
    ax_gain.set_xlabel("Time Step")
    ax_gain.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
    ax_gain.set_ylabel("$g_i$")

    ax_gain.set_title("B",{'fontweight' : 'bold'}, loc="left")

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

fig_gain, ax_gain = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

plot_gain(ax_gain)

fig_gain.tight_layout()
fig_gain.savefig("../../plots/gain.png", dpi=300)

plt.show()
