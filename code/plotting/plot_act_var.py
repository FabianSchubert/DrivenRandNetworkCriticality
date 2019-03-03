#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_act_var(ax_var,file="../../data/sim_results.npz"):

    Data = np.load(file)

    n_t = Data["n_t"]

    t_ax = np.array(range(n_t-100,n_t))

    ax_var.plot(Data["var_mean_rec"]/Data["std_act_target"]**2)
    ax_var.set_xlabel("Time Steps")
    ax_var.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)

    ax_var.set_ylabel("$\\mathrm{Var}\\left[ y_i \\right] / \\sigma^2_{t}$")
    #ax_var.grid()

    #ax_var.set_title("C",{'fontweight' : 'bold'}, loc="left")

if __name__=='__main__':

    textwidth = 5.5532
    std_figsize = (textwidth/2.,2.)
    dpi_screen = 120

    fig_var, ax_var = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

    plot_act_var(ax_var)

    fig_var.tight_layout()
    fig_var.savefig("../../plots/var.png", dpi=300)

    plt.show()
