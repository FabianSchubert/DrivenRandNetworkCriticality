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


n_samples_global = 3
n_samples_local = 1

delta_a_RTRL_global = np.ndarray((n_samples_global,4,50))
delta_a_RTRL_local = np.ndarray((n_samples_local,4,50))

for k in range(n_samples_global):
    delta_a_RTRL_global[k,:,:] = np.load(data_path + "delta_a_local_mean_RTRL_global_sample_" + str(k) + ".npy")

for k in range(n_samples_local):
    delta_a_RTRL_local[k,:,:] = np.load(data_path + "delta_a_local_mean_RTRL_local_sample_" + str(k) + ".npy")

'''
delta_a_RTRL_global_sample_0 = np.load(data_path + "delta_a_local_mean_RTRL_global_sample_0.npy")
delta_a_RTRL_global_sample_1 = np.load(data_path + "delta_a_local_mean_RTRL_global_sample_1.npy")
delta_a_RTRL_global_sample_2 = np.load(data_path + "delta_a_local_mean_RTRL_global_sample_1.npy")

delta_a_RTRL_local_sample_0 = np.load(data_path + "delta_a_local_mean_RTRL_local_sample_0.npy")
#delta_a_RTRL_local_sample_1 = np.load(data_path + "delta_a_local_mean_RTRL_local_sample_1.npy")



delta_a_RTRL_global[0,:,:] = delta_a_RTRL_global_sample_0
delta_a_RTRL_global[1,:,:] = delta_a_RTRL_global_sample_1

delta_a_RTRL_local[0,:,:] = delta_a_RTRL_local_sample_0
#delta_a_RTRL_local[1,:,:] = delta_a_RTRL_local_sample_1
'''


gain_arr = np.linspace(0.2,1.5,delta_a_RTRL_global.shape[2])
tau_arr = np.array([1,5,10,15])

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(2,2,figsize=std_figsize)

msqe = np.load(data_path + "msqe_sweep.npy")
#msqe = np.load(data_path + "msqe_sweep_small_network.npy")

ax_msqe = np.ndarray((2,2)).astype(type(ax[0,0]))

#fig.suptitle("Local Gain Adaptation",fontsize=12)



for k in range(4):


    #if k!=0:
    #    ax[int(k/2),k%2].get_legend().set_visible(False)

    #sns.lineplot(x="$a$",y="$\\Delta a$",data=df_global,ax=ax[int(k/2),k%2])



    #import pdb
    #pdb.set_trace()

    #'''
    mean_RTRL_local = delta_a_RTRL_local[:,k,:].mean(axis=0)
    mean_RTRL_global = delta_a_RTRL_global[:,k,:].mean(axis=0)
    mean_msqe = msqe[:,k,:].mean(axis=0)

    err_RTRL_local = delta_a_RTRL_local[:,k,:].std(axis=0)/delta_a_RTRL_local.shape[0]**.5
    err_RTRL_global = delta_a_RTRL_global[:,k,:].std(axis=0)/delta_a_RTRL_global.shape[0]**.5
    err_msqe = msqe[:,k,:].std(axis=0)/msqe.shape[0]**.5

    idx_local = np.where(np.abs(mean_RTRL_local) < 1.)
    idx_global = np.where(np.abs(mean_RTRL_global) < 1.)

    mean_RTRL_local = mean_RTRL_local[idx_local]
    mean_RTRL_global = mean_RTRL_global[idx_global]

    err_RTRL_local = err_RTRL_local[idx_local]
    err_RTRL_global = err_RTRL_global[idx_global]


    err_RTRL_local /= mean_RTRL_local.std()
    err_RTRL_global /= mean_RTRL_global.std()
    #err_msqe /= mean_msqe.std()

    mean_RTRL_local /= mean_RTRL_local.std()
    mean_RTRL_global /= mean_RTRL_global.std()
    #mean_msqe /= mean_msqe.std()


    ax[int(k/2),k%2].fill_between(gain_arr[idx_local],(mean_RTRL_local - err_RTRL_local),(mean_RTRL_local + err_RTRL_local),
                                  color=colors[0],alpha=0.4)
    if k == 0:
        ln1 = ax[int(k/2),k%2].plot(gain_arr[idx_local],mean_RTRL_local,c=colors[0],label="appr. local RTRL")
    else:
        ax[int(k/2),k%2].plot(gain_arr[idx_local],mean_RTRL_local,c=colors[0],label="appr. local RTRL")



    ax[int(k/2),k%2].fill_between(gain_arr[idx_global],(mean_RTRL_global - err_RTRL_global),(mean_RTRL_global + err_RTRL_global),
                                  color=colors[1],alpha=0.4)
    if k==0:
        ln2 = ax[int(k/2),k%2].plot(gain_arr[idx_global],mean_RTRL_global,c=colors[1],label="full RTRL")
    else:
        ax[int(k/2),k%2].plot(gain_arr[idx_global],mean_RTRL_global,c=colors[1],label="full RTRL")



    ax_msqe[int(k/2),k%2] = ax[int(k/2),k%2].twinx()



    ax_msqe[int(k/2),k%2].fill_between(gain_arr[idx_global],(mean_msqe - err_msqe),(mean_msqe + err_msqe),
                                  color=colors[2],alpha=0.4)
    if k == 0:
        ln3 = ax_msqe[int(k/2),k%2].plot(gain_arr[idx_global],mean_msqe,c=colors[2],label="MSE")
    else:
        ax_msqe[int(k/2),k%2].plot(gain_arr[idx_global],mean_msqe,c=colors[2],label="MSE")


    ax[int(k/2),k%2].set_title("$\\tau = " + str(tau_arr[k]) + "$")

    #ax[int(k/2),k%2].grid(False)
    ax_msqe[int(k/2),k%2].grid(False)
    #'''

ax[0,0].set_ylabel("$\\Delta a$")
ax[1,0].set_ylabel("$\\Delta a$")

ax[0,1].set_ylabel("")
ax[1,1].set_ylabel("")

ax[1,0].set_xlabel("$a$")
ax[1,1].set_xlabel("$a$")

ax[0,0].set_xlabel("")
ax[0,1].set_xlabel("")



ax_msqe[0,1].set_ylabel("MSE")
ax_msqe[1,1].set_ylabel("MSE")


lns = ln1 + ln2 + ln3
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

fig.savefig(plot_path + "delta_a_local.pdf")
fig.savefig(plot_path + "delta_a_local.png",dpi=600)

#fig.savefig(plot_path + "delta_a_local_small_network.pdf")
#fig.savefig(plot_path + "delta_a_local_small_network.png",dpi=600)

plt.show()
