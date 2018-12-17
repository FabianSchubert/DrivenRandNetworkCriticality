#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt



#================================
def plot_max_l_sweep(ax_max_l_sweep):

    Data = np.load("../../data/max_lyap_sweep/sim_results.npz")

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm = ax_max_l_sweep.pcolormesh(std_act_target_sweep_range,std_in_sweep_range,np.log(max_l))
    plt.colorbar(mappable=pcm,ax=ax_max_l_sweep)
    ax_max_l_sweep.contour(std_act_target_sweep_range,std_in_sweep_range,np.log(max_l), [0.],colors='r')

    ax_max_l_sweep.set_xlabel("$\\sigma$ act. target")
    ax_max_l_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_max_l_sweep.set_title("Max. Lyapunov Exp.")

#################################

#================================
def plot_gain_mean_sweep(ax_gain_mean_sweep):

    Data = np.load("../../data/max_lyap_sweep/sim_results.npz")

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]


    pcm2 = ax_gain_mean_sweep.pcolormesh(std_act_target_sweep_range,std_in_sweep_range,gain.mean(axis=2))
    plt.colorbar(mappable=pcm2,ax=ax_gain_mean_sweep)
    ax_gain_mean_sweep.contour(std_act_target_sweep_range,std_in_sweep_range,gain.mean(axis=2),cmap="inferno")

    ax_gain_mean_sweep.set_xlabel("$\\sigma$ act. target")
    ax_gain_mean_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_gain_mean_sweep.set_title("Pop. Mean of Gain")


#################################

#================================
def plot_gain_std_sweep(ax_gain_std_sweep):

    Data = np.load("../../data/max_lyap_sweep/sim_results.npz")

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm3 = ax_gain_std_sweep.pcolormesh(std_act_target_sweep_range,std_in_sweep_range,gain.std(axis=2))
    plt.colorbar(mappable=pcm3,ax=ax_gain_std_sweep)
    ax_gain_std_sweep.contour(std_act_target_sweep_range,std_in_sweep_range,gain.std(axis=2),cmap="inferno")

    ax_gain_std_sweep.set_xlabel("$\\sigma$ act. target")
    ax_gain_std_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_gain_std_sweep.set_title("Pop. Std. Dev. of Gain")


#################################

#================================
def plot_3d_gain_mean_sweep(ax_3d):

    Data = np.load("../../data/max_lyap_sweep/sim_results.npz")

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    STD_TARG, STD_IN = np.meshgrid(std_act_target_sweep_range,std_in_sweep_range)

    g_pred = (1.+STD_IN**2/STD_TARG**2)**(-.5)

    ax_3d.view_init(elev=20., azim=135.)

    ax_3d.plot_surface(STD_TARG,STD_IN,g_pred,label="Approximation")
    ax_3d.plot_surface(STD_TARG,STD_IN,gain.mean(axis=2),label="Simulation",alpha=0.5)

    ax_3d.set_xlabel("$\\sigma$ act. target")
    ax_3d.set_ylabel("$\\sigma$ ext. input")
    ax_3d.set_zlabel("$g$")

#################################
#ax_3d.legend()

if __name__ == "__main__":

    plt.style.use('matplotlibrc')

    textwidth = 5.5532
    std_figsize = (textwidth,4.)
    dpi_screen = 120


    fig_max_l_sweep, ax_max_l_sweep = plt.subplots(1,1,figsize=std_figsize, dpi=dpi_screen)

    plot_max_l_sweep(ax_max_l_sweep)

    fig_max_l_sweep.tight_layout()
    fig_max_l_sweep.savefig("../../plots/max_l_sweep.png", dpi=300)

    fig_gain_mean_sweep, ax_gain_mean_sweep = plt.subplots(1,1,figsize=std_figsize, dpi=dpi_screen)

    plot_gain_mean_sweep(ax_gain_mean_sweep)

    fig_gain_mean_sweep.tight_layout()
    fig_gain_mean_sweep.savefig("../../plots/gain_mean_sweep.png", dpi=300)

    fig_gain_std_sweep, ax_gain_std_sweep = plt.subplots(1,1,figsize=std_figsize, dpi=dpi_screen)

    plot_gain_std_sweep(ax_gain_std_sweep)

    fig_gain_std_sweep.tight_layout()
    fig_gain_std_sweep.savefig("../../plots/gain_std_sweep.png", dpi=300)

    from mpl_toolkits.mplot3d import Axes3D
    fig_3d = plt.figure(figsize=std_figsize, dpi=dpi_screen)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    plot_3d_gain_mean_sweep(ax_3d)

    fig_3d.tight_layout()
    fig_3d.savefig("../../plots/gain_std_sweep_3d.png", dpi=300)

plt.show()
