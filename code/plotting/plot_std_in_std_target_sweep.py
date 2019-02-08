#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.integrate import romberg
from scipy.integrate import quadrature
from scipy.optimize import root


# ================================
def plot_max_l_sweep(ax_max_l_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm = ax_max_l_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, np.log(max_l))
    plt.colorbar(mappable=pcm, ax=ax_max_l_sweep)
    '''
    ax_max_l_sweep.contour(std_act_target_sweep_range,
                           std_in_sweep_range, np.log(max_l), [0.], linewidths=2, linestyles="dashed",colors='#0000FF')
    '''
    ax_max_l_sweep.set_xlabel("$\\sigma$ act. target")
    ax_max_l_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_max_l_sweep.set_title("Max. Lyapunov Exp.")

#################################

#================================
def plot_max_l_crit_trans_sweep(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, np.log(max_l), [0.], linewidths=2, linestyles="dashed",colors=color)
    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")
#################################


# ================================


def plot_gain_mean_sweep(ax_gain_mean_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):
    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm2 = ax_gain_mean_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, gain.mean(axis=2))
    plt.colorbar(mappable=pcm2, ax=ax_gain_mean_sweep)
    #ax_gain_mean_sweep.contour(
    #    std_act_target_sweep_range, std_in_sweep_range, gain.mean(axis=2), cmap="inferno")

    '''
    ax_gain_mean_sweep.contour(std_act_target_sweep_range,
                           std_in_sweep_range, gain.mean(axis=2), [1.], linewidths=2, linestyles="dashed",colors='#0000FF')

    '''
    ax_gain_mean_sweep.set_xlabel("$\\sigma$ act. target")
    ax_gain_mean_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_gain_mean_sweep.set_title("Pop. Mean of Gain")


#################################

def plot_gain_mean_crit_trans_sweep(ax,critval=1.,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]
    ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, gain.mean(axis=2), [critval], linewidths=2, linestyles="dashed",colors=color)

    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")
# ================================
def plot_gain_std_sweep(ax_gain_std_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm3 = ax_gain_std_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, gain.std(axis=2))
    plt.colorbar(mappable=pcm3, ax=ax_gain_std_sweep)
    ax_gain_std_sweep.contour(std_act_target_sweep_range,
                              std_in_sweep_range, gain.std(axis=2), cmap="inferno")

    ax_gain_std_sweep.set_xlabel("$\\sigma$ act. target")
    ax_gain_std_sweep.set_ylabel("$\\sigma$ ext. input")

    #ax_gain_std_sweep.set_title("Pop. Std. Dev. of Gain")


#################################

# ================================
def plot_3d_gain_mean_sweep(ax_3d,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    STD_TARG, STD_IN = np.meshgrid(
        std_act_target_sweep_range, std_in_sweep_range)

    g_pred = (1. + STD_IN**2 / STD_TARG**2)**(-.5)

    ax_3d.view_init(elev=20., azim=135.)

    ax_3d.plot_surface(STD_TARG, STD_IN, g_pred, label="Approximation")
    ax_3d.plot_surface(STD_TARG, STD_IN, gain.mean(
        axis=2), label="Simulation", alpha=0.5)

    ax_3d.set_xlabel("$\\sigma$ act. target")
    ax_3d.set_ylabel("$\\sigma$ ext. input")
    ax_3d.set_zlabel("$g$")

#################################

#================================
def plot_echo_state_prop_trans(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):
    Data = np.load(file_path)
    esp = Data["echo_state_prop_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]
    ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, esp, [.5], linewidths=2, linestyles="dashed",colors=color)

    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")


#################################

# ax_3d.legend()


def sigm_int_func(x, g, sigm_ext, sigm_targ):
    return gauss(x, 0., (sigm_ext**2 + sigm_targ**2)**.5) * tanh_func(g * x)**2


def f_gain_root(x, sigm_ext, sigm_targ):
    int, err = quad(sigm_int_func, -np.infty, np.infty,
                    args=(x, sigm_ext, sigm_targ,), epsrel=0.000001)
    #int = romberg(sigm_int_func,-10.,10.,args=(x,sigm_ext,sigm_targ))
    #int, err = quadrature(sigm_int_func,-10.,10.,args=(x,sigm_ext,sigm_targ))
    return int - sigm_targ**2


def find_consist_gain(sigm_ext, sigm_targ):
    sol = root(f_gain_root, (1. + sigm_ext**2 / sigm_targ**2) **
               (-.5), (sigm_ext, sigm_targ))
    return sol['x']


def gauss(x, m, s):
    return np.exp(-(x - m)**2 / (2. * s**2)) / (2. * np.pi * s**2)**.5


def tanh_func(x):
    # return x
    # return x - x**3/3.
    return np.tanh(x)

# ================================


def plot_3d_gain_mean_sweep_full_tanh_pred(ax_3d,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    max_l = Data["max_l_list"]
    gain = Data["gain_list"]
    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    STD_TARG, STD_IN = np.meshgrid(
        std_act_target_sweep_range, std_in_sweep_range)

    g_pred = np.ndarray((n_sweep_std_in, n_sweep_std_act_target))

    for k in range(n_sweep_std_in):
        for l in range(n_sweep_std_act_target):
            g_pred[k, l] = find_consist_gain(
                std_in_sweep_range[k], std_act_target_sweep_range[l])

    #g_pred = (1.+STD_IN**2/STD_TARG**2)**(-.5)

    ax_3d.view_init(elev=20., azim=135.)

    ax_3d.plot_surface(STD_TARG, STD_IN, g_pred, label="Approximation")
    ax_3d.plot_surface(STD_TARG, STD_IN, gain.mean(
        axis=2), label="Simulation", alpha=0.5)

    ax_3d.set_xlabel("$\\sigma$ act. target")
    ax_3d.set_ylabel("$\\sigma$ ext. input")
    ax_3d.set_zlabel("$g$")
    '''
    fig, ax = plt.subplots(1,1)

    ax.plot(g_pred[5,:],c="b")
    ax.plot(gain.mean(axis=2)[5,:],c="b")

    ax.plot(g_pred[15,:],c="r")
    ax.plot(gain.mean(axis=2)[15,:],c="r")
    '''


#################################


# ================================
def plot_rmsqe_hom(ax,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    msqe = Data["trail_av_hom_error_list"]

    std_in_sweep_range = Data["std_in_sweep_range"]
    std_act_target_sweep_range = Data["std_act_target_sweep_range"]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    pcm = ax.pcolormesh(std_act_target_sweep_range,
                        std_in_sweep_range, msqe**.5)
    plt.colorbar(mappable=pcm, ax=ax)

    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")

#################################

# ================================
def plot_mem_cap(ax,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    mem_cap = Data["mem_cap_list"]

    std_e = Data["std_in_sweep_range"]
    std_act_t = Data["std_act_target_sweep_range"]

    pcm = ax.pcolormesh(std_act_t,std_e, mem_cap)

    plt.colorbar(mappable=pcm, ax=ax)

    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")


##################################

def plot_mem_cap_max_fixed_ext(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file_path)

    mem_cap = Data["mem_cap_list"]

    std_e = Data["std_in_sweep_range"]
    std_act_t = Data["std_act_target_sweep_range"]

    max_std_targ = np.ndarray((std_e.shape[0]-1))

    for k in range(1,std_e.shape[0]):
        max_std_targ[k-1] = std_act_t[np.argmax(mem_cap[k,:])]

    ax.plot(max_std_targ,std_e[1:],c=color)

    ax.set_xlabel("$\\sigma$ act. target")
    ax.set_ylabel("$\\sigma$ ext. input")



if __name__ == "__main__":

    plt.style.use('matplotlibrc')

    textwidth = 5.5532
    std_figsize = (textwidth, 4.)
    dpi_screen = 120

    fig_max_l_sweep, ax_max_l_sweep = plt.subplots(
        1, 1, figsize=std_figsize, dpi=dpi_screen)

    plot_max_l_sweep(ax_max_l_sweep)

    fig_max_l_sweep.tight_layout()
    fig_max_l_sweep.savefig("../../plots/max_l_sweep.png", dpi=300)

    fig_gain_mean_sweep, ax_gain_mean_sweep = plt.subplots(
        1, 1, figsize=std_figsize, dpi=dpi_screen)

    plot_gain_mean_sweep(ax_gain_mean_sweep)

    fig_gain_mean_sweep.tight_layout()
    fig_gain_mean_sweep.savefig("../../plots/gain_mean_sweep.png", dpi=300)

    fig_gain_std_sweep, ax_gain_std_sweep = plt.subplots(
        1, 1, figsize=std_figsize, dpi=dpi_screen)

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
