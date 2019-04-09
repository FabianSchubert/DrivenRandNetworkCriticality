#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')



def plot_gain_vs_sigm_ext_max_mc(ax,color="#0000FF",sigmw=1.,file="../../data/max_lyap_sweep/sim_results.npz"):

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



    gain = np.array([Dat["gain_list"] for Dat in Data]).mean(axis=3)

    mc = np.array([Dat["mem_cap_list"] for Dat in Data])

    gain_max_mc = np.ndarray((gain.shape[0],sigma_ext.shape[0]))

    for k in range(gain.shape[0]):
        gain_max_mc[k,:] = gain[k,range(sigma_ext.shape[0]),np.argmax(mc[k,:,:],axis=1)]

    gain_max_mc_mean = gain_max_mc.mean(axis=(0))
    gain_max_mc_err = gain_max_mc.std(axis=0)/gain_max_mc.shape[0]**.5

    #ind_argmax_sigma_t = np.ndarray((n_t),dtype="int")

    #for k in range(n_ext):
    #    ind_argmax_sigma_t[k] = np.argmax(mc[k,:])

    #gain_argmax_sigma_t = gain[range(n_ext),ind_argmax_sigma_t]

    #ax.plot(sigma_ext,gain_argmax_sigma_t)

    #ax.set_xlabel(r'$\sigma_{\rm ext}$')

    ax.plot(sigma_ext[1:],gain_max_mc_mean[1:],color=color,label='$\\sigma_{\\rm w} = $'+str(sigmw))
    ax.fill_between(sigma_ext[1:],gain_max_mc_mean[1:]-gain_max_mc_err[1:],gain_max_mc_mean[1:]+gain_max_mc_err[1:],color=color,alpha=0.3)

    ax.set_ylabel(r'$\left< a_i\right>_P\left\{ {\arg\, \max}_{\sigma_t} {\rm MC}\left( \sigma_t, \sigma_{\rm ext}\right), \sigma_{\rm ext} \right\}$')
    ax.set_xlabel(r'$\sigma_{\rm e} / \sigma_{\rm w}$')

def plot_fit_gain_vs_sigm_ext_max_mc(ax,file="../../data/max_lyap_sweep/sim_results.npz"):

    Data = np.load(file)

    sigma_ext = Data["std_in_sweep_range"]
    n_ext = sigma_ext.shape[0]
    sigma_t = Data["std_act_target_sweep_range"]
    n_t = sigma_t.shape[0]

    gain = Data["gain_list"].mean(axis=2)

    mc = Data["mem_cap_list"]

    ind_argmax_sigma_t = np.ndarray((n_t),dtype="int")

    for k in range(n_ext):
        ind_argmax_sigma_t[k] = np.argmax(mc[k,:])

    gain_argmax_sigma_t = gain[range(n_ext),ind_argmax_sigma_t]

    fit, res, rank, singv, rcond = np.polyfit(sigma_ext,gain_argmax_sigma_t,1,full=True)

    Rsqu = 1.-res/((gain_argmax_sigma_t-gain_argmax_sigma_t.mean())**2.).sum()

    ax.plot(sigma_ext,sigma_ext*fit[0] + fit[1],label="fit: ${0:4.3} + ".format(fit[1]) + "\\sigma_{\\rm ext}" + "{0:4.3}$".format(fit[0]))



if __name__ == '__main__':

    textwidth = 5.5532

    fig, ax = plt.subplots(1,figsize=(textwidth,0.6*textwidth))

    plot_gain_vs_sigm_ext_max_mc(ax)
    plot_fit_gain_vs_sigm_ext_max_mc(ax)

    ax.legend()

    fig.tight_layout()

    plt.show()


file = "../../data/max_lyap_sweep/sim_results.npz"
