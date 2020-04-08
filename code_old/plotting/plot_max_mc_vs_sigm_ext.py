#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

import matplotlib as mpl


def plot_max_mc_vs_sigm_ext(ax,color="#0000FF",sigmw=1.,file="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file,list):
        Data = [np.load(fl) for fl in file]
    else:
        Data = [np.load(file)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()


    sigma_ext = std_in_sweep_range[0]
    sigma_t = std_act_target_sweep_range[0]

    n_ext = sigma_ext.shape[0]
    n_t = sigma_t.shape[0]


    mc = np.array([Dat["mem_cap_list"] for Dat in Data])

    mc_max_sigm_e = mc.max(axis=2)

    mc_max_sigm_e_mean = mc_max_sigm_e.mean(axis=0)
    mc_max_sigm_e_err = mc_max_sigm_e.std(axis=0)/mc_max_sigm_e.shape[0]**.5

    ax.fill_between(sigma_ext/sigmw,mc_max_sigm_e_mean-mc_max_sigm_e_err,mc_max_sigm_e_mean+mc_max_sigm_e_err,alpha=0.5)
    ax.plot(sigma_ext/sigmw,mc_max_sigm_e_mean,color=color,label='$\\sigma_{\\rm w} = $'+str(sigmw))

    ax.set_xlabel(r'$\sigma_{\rm e} / \sigma_{\rm w}$')
    ax.set_ylabel(r'max. MC')



if __name__ == '__main__':

    textwidth = 5.5532

    figure_file = "../../plots/max_mc_vs_sigm_ext"

    output_formats = ['pdf','png']

    files1p0 = ["../../data/max_lyap_sweep/sim_results_mc_sigmaw_1p0_1.npz",
            "../../data/max_lyap_sweep/sim_results_mc_sigmaw_1p0_2.npz",
            "../../data/max_lyap_sweep/sim_results_mc_sigmaw_1p0_3.npz"]
    files2p0 = ["../../data/max_lyap_sweep/sim_results_mc_sigmaw_2p0_1.npz",
                "../../data/max_lyap_sweep/sim_results_mc_sigmaw_2p0_2.npz",
                "../../data/max_lyap_sweep/sim_results_mc_sigmaw_2p0_3.npz"]
    files0p5 = ["../../data/max_lyap_sweep/sim_results_mc_sigmaw_0p5_1.npz",
                "../../data/max_lyap_sweep/sim_results_mc_sigmaw_0p5_2.npz",
                "../../data/max_lyap_sweep/sim_results_mc_sigmaw_0p5_3.npz"]

    cmap = mpl.cm.get_cmap('viridis')

    fig, ax = plt.subplots(1,figsize=(textwidth,0.6*textwidth))

    plot_max_mc_vs_sigm_ext(ax,color=cmap(0.),sigmw=.5,file=files0p5)
    plot_max_mc_vs_sigm_ext(ax,color=cmap(.5),sigmw=1.,file=files1p0)
    plot_max_mc_vs_sigm_ext(ax,color=cmap(1.),sigmw=2.,file=files2p0)

    ax.legend()

    fig.tight_layout(pad=0.1)

    for format in output_formats:
        if format=="png":
            fig.savefig(figure_file+"."+format, dpi=1000)
        else:
            fig.savefig(figure_file+"."+format)

    plt.show()
