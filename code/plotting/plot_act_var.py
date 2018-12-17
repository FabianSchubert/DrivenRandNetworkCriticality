#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_act_var(ax_var):

    Data = np.load("../../data/sim_results.npz")

    n_t = Data["n_t"]

    t_ax = np.array(range(n_t-100,n_t))

    ax_var.plot(Data["var_mean_rec"])
    ax_var.set_xlabel("Time Step")
    ax_var.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
    ax_var.set_ylabel("$\\langle \\left( x_i^t - \\langle x_i \\rangle \\right)^2\\rangle_{\\rm pop}$")

    ax_var.grid()

    ax_var.set_title("C",{'fontweight' : 'bold'}, loc="left")

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

fig_var, ax_var = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

plot_act_var(ax_var)

fig_var.tight_layout()
fig_var.savefig("../../plots/var.png", dpi=300)

plt.show()
