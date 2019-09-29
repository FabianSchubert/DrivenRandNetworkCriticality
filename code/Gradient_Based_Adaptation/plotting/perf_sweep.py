#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter
sns.set()

mpl.style.use("matplotlibrc")

perf_gauss = np.load("../../../data/perf_xor_gaussian.npy")
specrad_gauss = np.load("../../../data/specrad_gaussian.npy")

perf_bin = np.load("../../../data/perf_xor_binary.npy")
specrad_bin = np.load("../../../data/specrad_binary.npy")

perf_gauss_smooth = gaussian_filter(perf_gauss,sigma=1)
perf_bin_smooth = gaussian_filter(perf_bin,sigma=1)

argmax_perf_gauss = np.argmax(perf_gauss_smooth,axis=1)
argmax_perf_bin = np.argmax(perf_bin_smooth,axis=1)

sigm_e = np.linspace(0.,1.5,31)
sigm_t = np.linspace(0.,0.9,31)

textwidth=6.3

fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))

pc_0 = ax[0].pcolormesh(sigm_t-0.5*(sigm_t[1]-sigm_t[0]),sigm_e-0.5*(sigm_e[1]-sigm_e[0]),perf_gauss)
plt.colorbar(ax=ax[0],mappable=pc_0)
ax[0].contour(sigm_t[:-1],sigm_e[:-1],specrad_gauss,levels=[1.],linewidths=[2],linestyles='dashed',colors=['b'])
ax[0].plot(sigm_t[argmax_perf_gauss[1:]],sigm_e[1:-1],lw=2,c=(1.,.7,0.))

pc_1 = ax[1].pcolormesh(sigm_t-0.5*(sigm_t[1]-sigm_t[0]),sigm_e-0.5*(sigm_e[1]-sigm_e[0]),perf_bin)
plt.colorbar(ax=ax[1],mappable=pc_1)
ax[1].contour(sigm_t[:-1],sigm_e[:-1],specrad_bin,levels=[1.],linewidths=[2],linestyles='dashed',colors=['b'])
ax[1].plot(sigm_t[argmax_perf_bin[1:]],sigm_e[1:-1],lw=2,c=(1.,.7,0.))

ax[0].set_xlabel('$\\sigma_{\\rm t}$')
ax[1].set_xlabel('$\\sigma_{\\rm t}$')
ax[0].set_ylabel('$\\sigma_{\\rm e}$')

fig.tight_layout(pad=0.1)

fig.savefig("../../../plots/Gradient_Based_Adaptation/perf_sweep.pdf")
fig.savefig("../../../plots/Gradient_Based_Adaptation/perf_sweep.png",dpi=1000)

plt.show()
