#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
plt.style.use('matplotlibrc')


style = sns.axes_style()

style["grid.color"] = '1.'


sns.set_style(style)



textwidth = 5.5532
std_figsize = (textwidth,.7*textwidth)
dpi_screen = 120

data_path = "../../../data/Gradient_Based_Adaptation/"
plot_path = "../../../plots/Gradient_Based_Adaptation/"

size_list = [1000,500,200,50]

msqe = []

for N in size_list:
    msqe.append(np.load(data_path + "msqe_sweep_N_" + str(N) + ".npy"))


gain_arr = np.linspace(0.2,1.5,msqe[0].shape[2])
tau_arr = np.array([1,5,10,15])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(2,2,figsize=std_figsize)

ln_msqe = []

for k in range(4):



    for n, N in enumerate(size_list):

        mean_msqe = msqe[n][:,k,:].mean(axis=0)
        err_msqe = msqe[n][:,k,:].std(axis=0)/msqe[n].shape[0]**.5

        ax[int(k/2),k%2].fill_between(gain_arr,(mean_msqe - err_msqe),(mean_msqe + err_msqe),
                                      color=colors[n],alpha=0.4)

        if k == 0:

            ln_msqe.append(ax[int(k/2),k%2].plot(gain_arr,mean_msqe,c=colors[n],label="$N = " + str(N) + "$"))
        else:
            ax[int(k/2),k%2].plot(gain_arr,mean_msqe,c=colors[n],label="$N = " + str(N) + "$")

    '''
    mean_msqe_medium = msqe_medium[:,k,:].mean(axis=0)
    mean_msqe_small = msqe_small[:,k,:].mean(axis=0)

    err_msqe = msqe[:,k,:].std(axis=0)/msqe.shape[0]**.5
    err_msqe_medium = msqe_medium[:,k,:].std(axis=0)/msqe_medium.shape[0]**.5
    err_msqe_small = msqe_small[:,k,:].std(axis=0)/msqe_small.shape[0]**.5

    ax[int(k/2),k%2].fill_between(gain_arr,(mean_msqe - err_msqe),(mean_msqe + err_msqe),
                                  color=colors[0],alpha=0.4)
    ax[int(k/2),k%2].fill_between(gain_arr,(mean_msqe_medium - err_msqe_medium),(mean_msqe_medium + err_msqe_medium),
                                  color=colors[1],alpha=0.4)
    ax[int(k/2),k%2].fill_between(gain_arr,(mean_msqe_small - err_msqe_small),(mean_msqe_small + err_msqe_small),
                                  color=colors[2],alpha=0.4)

    if k == 0:
        ln_msqe = ax[int(k/2),k%2].plot(gain_arr,mean_msqe,c=colors[0],label="$N = 1000$")
        ln_msqe_medium = ax[int(k/2),k%2].plot(gain_arr,mean_msqe_medium,c=colors[1],label="$N = 200$")
        ln_msqe_small = ax[int(k/2),k%2].plot(gain_arr,mean_msqe_small,c=colors[2],label="$N = 50$")
    else:
        ax[int(k/2),k%2].plot(gain_arr,mean_msqe,c=colors[0],label="$N = 1000$")
        ax[int(k/2),k%2].plot(gain_arr,mean_msqe_medium,c=colors[1],label="$N = 200$")
        ax[int(k/2),k%2].plot(gain_arr,mean_msqe_small,c=colors[2],label="$N = 50$")

    '''
    ax[int(k/2),k%2].set_title("$\\tau = " + str(tau_arr[k]) + "$")

    #ax[int(k/2),k%2].grid(False)
    #ax_msqe[int(k/2),k%2].grid(False)
    #'''

ax[0,0].set_ylabel("MSE")
ax[1,0].set_ylabel("MSE")

ax[0,1].set_ylabel("")
ax[1,1].set_ylabel("")

ax[1,0].set_xlabel("$a$")
ax[1,1].set_xlabel("$a$")

ax[0,0].set_xlabel("")
ax[0,1].set_xlabel("")



lns = ln_msqe[0]
if len(ln_msqe) > 1:
    for ln in ln_msqe[1:]:
        lns += ln

#lns = ln_msqe + ln_msqe_medium + ln_msqe_small
labs = [l.get_label() for l in lns]

ax[0,0].legend(lns, labs)


'''
ax[0,0].set_ylim([-.04,.07])
ax[0,1].set_ylim([-.04,.07])
ax[1,0].set_ylim([-.04,.07])
ax[1,1].set_ylim([-.04,.07])
'''
'''
ax[0,0].set_ylim([-.7,1.5])
ax[0,1].set_ylim([-1.,1.5])
ax[1,0].set_ylim([-1.7,1.5])
ax[1,1].set_ylim([-3.2,1.5])
'''

fig.tight_layout(rect=[0.,0.,1.,1.],pad=.1,h_pad=.2,w_pad=.2)

fig.savefig(plot_path + "performance_network_size.pdf")
fig.savefig(plot_path + "performance_network_size.png",dpi=600)

#fig.savefig(plot_path + "delta_a_local_small_network.pdf")
#fig.savefig(plot_path + "delta_a_local_small_network.png",dpi=600)

plt.show()
