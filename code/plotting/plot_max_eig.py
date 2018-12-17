#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_max_eig(ax_eig):

    Data = np.load("../../data/sim_results.npz")

    n_t = Data["n_t"]
    gain_rec = Data["gain_rec"]
    W = Data["W"]

    t_ax = np.array(range(n_t-100,n_t))

    dt_subsample = 1000

    l_abs_max = np.ndarray((int(n_t/dt_subsample)))

    for k in tqdm(range(int(n_t/dt_subsample))):
        l_abs_max[k] = np.abs(np.linalg.eigvals((W.T*gain_rec[k*dt_subsample,:]).T)).max()



    t_ax_eig = np.array(range(int(n_t/dt_subsample)))*dt_subsample

    ax_eig.plot(t_ax_eig, np.log(l_abs_max))

    ax_eig.set_xlabel("Time Steps")
    ax_eig.set_ylabel("${\\rm log \\left(arg\\,max_i\\{||\\lambda_i^t||\\}\\right) }$")

    ax_eig.grid()

    ax_eig.set_title("D",{'fontweight' : 'bold'}, loc="left")

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

fig_eig, ax_eig = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

plot_max_eig(ax_eig)

fig_eig.tight_layout()
fig_eig.savefig("../../plots/eig.png", dpi=300)

plt.show()
