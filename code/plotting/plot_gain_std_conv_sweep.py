#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gain_std_conv(ax,file = "../../data/gain_conv_sweep.npz"):

    sns.set()
    plt.style.use("matplotlibrc")

    Data = np.load(file)

    N_net_range = Data["N_net_range"]
    gain_std = Data["gain_pop_std"]

    ax.plot(N_net_range,gain_std**2,'-o')

    ax.set_ylabel("$\\mathrm{Var}\\left[ a_i \\right]$")
    ax.set_xlabel("$N$")

    ax.set_xscale("log")
    ax.set_yscale("log")


if __name__ == "__main__":

    fig, ax = plt.subplots(1,1)

    plot_gain_std_conv(ax)
    fig.tight_layout()

    plt.show()
