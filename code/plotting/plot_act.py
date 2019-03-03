#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_act_pop_mean(ax_act,file="../../data/sim_results.npz"):

    Data = np.load(file)

    n_t = Data["n_t"]
    t_ext_off = Data["t_ext_off"]

    t_ax = np.array(range(n_t))


    ax_act.plot(t_ax,Data["x_net_rec"].mean(axis=1))
    ax_act.set_xlabel("Time Steps")
    ax_act.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
    ax_act.set_ylabel("$\\left\\langle y_i \\right\\rangle_P$")

    #ax_act_trans.set_title("A",{'fontweight' : 'bold'}, loc="left")

def plot_act_trail_av(ax_act,file="../../data/sim_results.npz"):

    Data = np.load(file)

    n_t = Data["n_t"]
    t_ax = np.array(range(n_t))

    y_trail_av = Data["x_net_trail_av_rec"]

    ax_act.plot(t_ax,y_trail_av[:,::10])
    ax_act.set_xlabel("Time Steps")
    ax_act.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
    ax_act.set_ylabel("$\\overline{y}_i$")


if __name__=='__main__':

    textwidth = 5.5532
    std_figsize = (textwidth/2.,2.)
    dpi_screen = 120

    #=====================================
    fig_act, ax_act = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

    plot_act_pop_mean(ax_act_trans)

    fig_act.tight_layout()
    fig_act.savefig("../../plots/act_pop_mean.png", dpi=300)

    plt.show()
