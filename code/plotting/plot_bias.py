#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_bias(ax_bias,file="../../data/sim_results.npz"):

    Data = np.load(file)

    n_t = Data["n_t"]
    t_ext_off = Data["t_ext_off"]

    t_ax = np.array(range(n_t-100,n_t))

    ax_bias.plot(Data["bias_rec"][:,::10])
    ax_bias.set_xlabel("Time Steps")
    ax_bias.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
    ax_bias.set_ylabel("$b_i$")

    #ax_bias.set_title("B",{'fontweight' : 'bold'}, loc="left")

if __name__=='__main__':

    textwidth = 5.5532
    std_figsize = (textwidth/2.,2.)
    dpi_screen = 120

    fig_bias, ax_bias = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

    plot_bias(ax_bias)

    fig_gain.tight_layout()
    fig_gain.savefig("../../plots/bias.png", dpi=300)

    plt.show()
