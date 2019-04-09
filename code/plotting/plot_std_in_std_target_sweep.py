#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.integrate import romberg
from scipy.integrate import quadrature
from scipy.optimize import root

from scipy.interpolate import make_interp_spline, BSpline, UnivariateSpline

import sys

import pdb

sigm_ext_label = "$\\sigma_{\\rm ext}$ (input)"
sigm_targ_label = "$\\sigma_{\\rm t}$ (target)"


# ================================
def plot_max_l_sweep(ax_max_l_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):



    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    max_l = [Dat["max_l_list"] for Dat in Data]



    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    max_l_mean = np.array(max_l).mean(axis=0)

    pcm = ax_max_l_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, np.log(max_l_mean),rasterized=True)
    cb = plt.colorbar(mappable=pcm, ax=ax_max_l_sweep)
    #cb.outline.set_visible(False)

    '''
    ax_max_l_sweep.contour(std_act_target_sweep_range,
                           std_in_sweep_range, np.log(max_l), [0.], linewidths=2, linestyles="dashed",colors='#0000FF')
    '''
    ax_max_l_sweep.set_xlabel(sigm_targ_label)
    ax_max_l_sweep.set_ylabel(sigm_ext_label)

    #ax_max_l_sweep.set_title("Max. Lyapunov Exp.")

#################################

#================================
def plot_max_l_crit_trans_sweep(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    max_l = [Dat["max_l_list"] for Dat in Data]



    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    #pdb.set_trace()

    max_l_mean = np.array(max_l).mean(axis=0)

    ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, np.log(max_l_mean), [0.], linewidths=2, linestyles="dashed",colors=color)
    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)
#################################


# ================================


def plot_gain_mean_sweep(ax_gain_mean_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):
    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    gain = [Dat["gain_list"] for Dat in Data]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array(gain)

    pcm2 = ax_gain_mean_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, gain.mean(axis=(0,3)),rasterized=True)
    cb=plt.colorbar(mappable=pcm2, ax=ax_gain_mean_sweep)
    #cb.outline.set_visible(False)
    #ax_gain_mean_sweep.contour(
    #    std_act_target_sweep_range, std_in_sweep_range, gain.mean(axis=2), cmap="inferno")

    '''
    ax_gain_mean_sweep.contour(std_act_target_sweep_range,
                           std_in_sweep_range, gain.mean(axis=2), [1.], linewidths=2, linestyles="dashed",colors='#0000FF')

    '''
    ax_gain_mean_sweep.set_xlabel(sigm_targ_label)
    ax_gain_mean_sweep.set_ylabel(sigm_ext_label)

    #ax_gain_mean_sweep.set_title("Pop. Mean of Gain")


#################################

def plot_gain_mean_crit_trans_sweep(ax,critval=1.,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    gain = [Dat["gain_list"] for Dat in Data]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array(gain)


    ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, gain.mean(axis=(0,3)), [critval], linewidths=2, linestyles="dashed",colors=color)

    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)
# ================================
def plot_gain_std_sweep(ax_gain_std_sweep,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    gain = [Dat["gain_list"] for Dat in Data]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array(gain)

    pcm3 = ax_gain_std_sweep.pcolormesh(
        std_act_target_sweep_range, std_in_sweep_range, gain.std(axis=(0,3)),rasterized=True)
    cb=plt.colorbar(mappable=pcm3, ax=ax_gain_std_sweep)
    #cb.outline.set_visible(False)

    ax_gain_std_sweep.contour(std_act_target_sweep_range,
                              std_in_sweep_range, gain.std(axis=2), cmap="inferno")

    ax_gain_std_sweep.set_xlabel(sigm_targ_label)
    ax_gain_std_sweep.set_ylabel(sigm_ext_label)

    #ax_gain_std_sweep.set_title("Pop. Std. Dev. of Gain")


#################################

# ================================
def plot_3d_gain_mean_sweep(ax_3d,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    gain = [Dat["gain_list"] for Dat in Data]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array(gain)

    STD_TARG, STD_IN = np.meshgrid(
        std_act_target_sweep_range, std_in_sweep_range)

    g_pred = (1. + STD_IN**2 / STD_TARG**2)**(-.5)

    ax_3d.view_init(elev=20., azim=135.)

    ax_3d.plot_surface(STD_TARG, STD_IN, g_pred, label="Approximation")
    ax_3d.plot_surface(STD_TARG, STD_IN, gain.mean(
        axis=(0,3)), label="Simulation", alpha=0.5)

    ax_3d.set_xlabel(sigm_targ_label)
    ax_3d.set_ylabel(sigm_ext_label)
    ax_3d.set_zlabel("$\\left\\langle a_i \\right\\rangle$")

#################################

#================================
def plot_2d_gain_mean_sweep(ax_2d,ind_std_ext,colorsim='#0000FF',colorpred='#FF0000',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    gain = [Dat["gain_list"] for Dat in Data]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array(gain)

    STD_TARG, STD_IN = np.meshgrid(
        std_act_target_sweep_range, std_in_sweep_range)

    g_pred = (1. + STD_IN**2 / STD_TARG**2)**(-.5)

    gain_mean = gain.mean(axis=(0,3))
    gain_err = gain.std(axis=(0,3))/(gain.shape[0]*gain.shape[3])**.5

    ax_2d.errorbar(STD_TARG[ind_std_ext,:],gain_mean[ind_std_ext,:],yerr=gain_err[ind_std_ext,:],fmt="^", markersize=5, label="Simulation",color=colorsim)
    #ax_2d.plot(STD_TARG[ind_std_ext,:], gain.mean(axis=2)[ind_std_ext,:], "^", markersize=5, label="Simulation",color=colorsim)
    #ax_2d.plot()
    ax_2d.plot(STD_TARG[ind_std_ext,:], g_pred[ind_std_ext,:], label="Analytic Prediction",color=colorpred)


    ax_2d.set_xlabel(sigm_targ_label)
    ax_2d.set_ylabel("$\\left\\langle a_i \\right\\rangle_P$")



#################################


#================================
def plot_echo_state_prop_trans(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    esp = [Dat["echo_state_prop_list"] for Dat in Data]

    esp_mean = np.array(esp).mean(axis=0)

    cont = ax.contour(std_act_target_sweep_range,
                           std_in_sweep_range, esp_mean, [.5], linewidths=0, linestyles="dashed",colors=color)

    p=cont.collections[0].get_paths()[0]
    v = p.vertices
    x_cont = v[:,0]
    y_cont = v[:,1]

    t_param = np.linspace(0.,1.,x_cont.shape[0])

    t_param_smooth = np.linspace(0.,1.,1000)

    #y = np.linspace(std_in_sweep_range[0],std_in_sweep_range[-1],1000)

    #spl = make_interp_spline(std_e[1:],max_std_targ, k=n_deg)
    spl_x = UnivariateSpline(t_param,x_cont,s=.002)
    spl_y = UnivariateSpline(t_param,y_cont,s=.002)

    #pfit = np.flip(np.polyfit(max_std_targ,std_e[1:],n_deg))

    x = spl_x(t_param_smooth)
    y = spl_y(t_param_smooth)


    #x = np.linspace(0.1,std_act_target_sweep_range[-1],1000)

    #n_deg = 6

    #pfit = np.flip(np.polyfit(x_cont,y_cont,n_deg))

    #y = np.zeros((1000))

    #for k in range(n_deg+1):
    #   y += pfit[k] * x**k

    pfplot, = ax.plot(x,y,linestyle=(0.,(.4, .4)),linewidth=3,color=color)

    #import pdb; pdb.set_trace()


    #cont.collections[0].set_linestyles([(0.0,[.4,.4])])

    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)


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

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    gain = np.array([Dat["gain_list"] for Dat in Data])

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
        axis=(0,3)), label="Simulation", alpha=0.5)

    ax_3d.set_xlabel(sigm_targ_label)
    ax_3d.set_ylabel(sigm_ext_label)
    ax_3d.set_zlabel("$\\left\\langle a_i \\right\\rangle$")
    '''
    fig, ax = plt.subplots(1,1)

    ax.plot(g_pred[5,:],c="b")
    ax.plot(gain.mean(axis=2)[5,:],c="b")

    ax.plot(g_pred[15,:],c="r")
    ax.plot(gain.mean(axis=2)[15,:],c="r")
    '''


#################################

#================================
def plot_2d_gain_mean_sweep_full_tanh_pred(ax_2d,ind_std_ext,colorsim='#0000FF',colorpred='#FF0000',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_in_sweep_range = std_in_sweep_range[0]
    std_act_target_sweep_range = std_act_target_sweep_range[0]

    n_sweep_std_in = std_in_sweep_range.shape[0]
    n_sweep_std_act_target = std_act_target_sweep_range.shape[0]

    STD_TARG, STD_IN = np.meshgrid(
        std_act_target_sweep_range, std_in_sweep_range)

    gain = np.array([Dat["gain_list"] for Dat in Data])

    g_pred = np.ndarray((n_sweep_std_act_target))

    for l in range(n_sweep_std_act_target):
        g_pred[l] = find_consist_gain(
            std_in_sweep_range[ind_std_ext], std_act_target_sweep_range[l])

    ax_2d.plot(STD_TARG[ind_std_ext,:], gain.mean(axis=(0,3))[ind_std_ext,:], "^", markersize=5, label="Simulation",color=colorsim)
    ax_2d.plot(STD_TARG[ind_std_ext,:], g_pred, label="Numeric Prediction",color=colorpred)


    ax_2d.set_xlabel(sigm_targ_label)
    ax_2d.set_ylabel("$\\left\\langle a_i \\right\\rangle_{\\rm P}$")



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
                        std_in_sweep_range, msqe**.5,rasterized=True)
    cb=plt.colorbar(mappable=pcm, ax=ax)
    #cb.outline.set_visible(False)

    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)

#################################

# ================================
def plot_mem_cap(ax,file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_e = std_in_sweep_range[0]
    std_act_t = std_act_target_sweep_range[0]

    n_sweep_std_in = std_e.shape[0]
    n_sweep_std_act_target = std_e.shape[0]

    mem_cap = np.array([Dat["mem_cap_list"] for Dat in Data])

    pcm = ax.pcolormesh(std_act_t,std_e, mem_cap.mean(axis=0),rasterized=True)

    cb=plt.colorbar(mappable=pcm, ax=ax)
    #cb.outline.set_visible(False)

    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)


##################################

def plot_mem_cap_max_fixed_ext(ax,color='#0000FF',file_path="../../data/max_lyap_sweep/sim_results.npz"):

    if isinstance(file_path, list):
        Data = [np.load(file) for file in file_path]
    else:
        Data = [np.load(file_path)]

    std_in_sweep_range = [Dat["std_in_sweep_range"] for Dat in Data]
    std_act_target_sweep_range = [Dat["std_act_target_sweep_range"] for Dat in Data]

    for k in range(1,len(Data)):
        if not(np.array_equal(std_in_sweep_range[k],std_in_sweep_range[k-1]) and np.array_equal(std_act_target_sweep_range[k],std_act_target_sweep_range[k-1])):
            print("Data dimensions do not fit!")
            sys.exit()

    std_e = std_in_sweep_range[0]
    std_act_t = std_act_target_sweep_range[0]

    n_sweep_std_in = std_e.shape[0]
    n_sweep_std_act_target = std_act_t.shape[0]

    mem_cap = np.array([Dat["mem_cap_list"] for Dat in Data])

    #max_std_targ = np.ndarray((std_e.shape[0]-1))

    max_std_targ = np.ndarray((mem_cap.shape[0],std_e.shape[0]-1,))

    for k in range(mem_cap.shape[0]):
        max_std_targ[k,:] = std_act_t[np.argmax(mem_cap[k,:,:],axis=1)][1:]

    max_std_targ_mean = max_std_targ.mean(axis=0)
    max_std_targ_err = max_std_targ.std(axis=0)/max_std_targ.shape[0]**.5

    #for k in range(1,std_e.shape[0]):
    #    max_std_targ[k-1] = std_act_t[np.argmax(mem_cap[k,:])]

    y = np.linspace(std_e[1],std_e[-1],1000)

    #spl = make_interp_spline(std_e[1:],max_std_targ, k=n_deg)
    spl_mean = UnivariateSpline(std_e[1:],max_std_targ_mean,s=.002)
    spl_err_low = UnivariateSpline(std_e[1:],max_std_targ_mean-max_std_targ_err,s=.002)
    spl_err_high = UnivariateSpline(std_e[1:],max_std_targ_mean+max_std_targ_err,s=.002)
    #pfit = np.flip(np.polyfit(max_std_targ,std_e[1:],n_deg))

    x_m = spl_mean(y)
    x_err_low = spl_err_low(y)
    x_err_high = spl_err_high(y)

    #for k in range(n_deg+1):
    #   y += pfit[k] * x**k

    pfplot, = ax.plot(x_m,y,linewidth=1,color=color)
    ax.fill_betweenx(y,x_err_low,x_err_high,color=color,alpha=0.5)

    #ax.plot(max_std_targ,std_e[1:],c=color,linewidth=2)

    ax.set_xlabel(sigm_targ_label)
    ax.set_ylabel(sigm_ext_label)

    ax.set_xlim([std_act_t[0],std_act_t[-1]])
    ax.set_ylim([std_e[1],std_e[-1]])


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
