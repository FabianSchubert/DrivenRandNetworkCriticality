#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('matplotlibrc')

Data = np.load("../../data/sim_results.npz")

n_t = Data["n_t"]
gain_rec = Data["gain_rec"]

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

fig_gain_dist, ax_gain_dist = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

ax_gain_dist.hist(gain_rec[-1,:],bins=30,density=True)

ax_gain_dist.set_xlim([0.,1.5])

ax_gain_dist.set_xlabel("gain")
ax_gain_dist.set_ylabel("prob.")

fig_gain_dist.tight_layout()
fig_gain_dist.savefig("../../plots/gain_dist.png", dpi=300)


plt.show()
