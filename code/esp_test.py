#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net_simple as dns

import seaborn as sns
sns.set()

import copy

from scipy.optimize import curve_fit

from tqdm import tqdm

def f_piecew_lin(x,y_0,a,x_0):
    return (x<=x_0)*(y_0+a*x) + (x>x_0)*(y_0+a*x_0)

def fit_piecew_lin(logd):
    t = np.array(range(logd.shape[0]))

    y_0_init = logd[0]
    a_init = (logd[100]-logd[0])/100.
    x_0_init = 0.75*(logd[-1]-y_0_init)/a_init

    fitp,fitcov = curve_fit(f_piecew_lin,t,logd,[y_0_init,a_init,x_0_init])

    return fitp, fitcov

def act_f(x):
    return (1.-np.exp(-x**2.))**.5*x/np.abs(x)

def run_esp_test(N_net_test,std_in_test,gain_test,n_t_test,d_offs):

    Net1 = dns(N_net=N_net_test,std_in=std_in_test,gain=gain_test,n_t=n_t_test,rand_inp_seed=np.random.randint(100000))
    #Net1.s = act_f
    Net2 = copy.deepcopy(Net1)

    rand_offs = np.random.normal(0.,1.,(Net2.N_net))
    rand_offs *= d_offs/np.linalg.norm(rand_offs)

    Net2.x_net += rand_offs

    Net1.run_sim()
    Net2.run_sim()

    d = np.linalg.norm(Net1.x_net_rec-Net2.x_net_rec,axis=1)
    logd = np.log10(d)

    fitp,fitcov = fit_piecew_lin(logd)

    return Net1, Net2, d, logd, [fitp,fitcov]


n_sweep = 5

conv_rate_sweep = np.ndarray((n_sweep))
conv_rate_approx_sweep = np.ndarray((n_sweep))

std_in_sweep = np.linspace(0.,.2,n_sweep)

for k in tqdm(range(n_sweep)):

    Net1, Net2, d, logd, [p,cov] = run_esp_test(1000,std_in_sweep[k],1.3,5000,0.01)
    #import pdb; pdb.set_trace()

    var_total = np.reshape(Net1.W @ Net1.x_net_rec.T + Net1.I_in_rec.T,(Net1.n_t*Net1.N_net)).var()

    conv_rate_approx_sweep[k] = np.log10(Net1.gain[0]*Net1.std_conn**.5/(1.+4.*Net1.gain[0]**2.*var_total)**.25)

    conv_rate_sweep[k] = p[1]

'''
fig_x, ax_x = plt.subplots()

ax_x.plot(Net1.x_net_rec[:,0])
ax_x.plot(Net2.x_net_rec[:,0])

t = np.array(range(Net1.n_t))

fig_d, ax_d = plt.subplots()

ax_d.plot(t,d)
ax_d.plot(t,10.**f_piecew_lin(t,p[0],p[1],p[2]))
ax_d.set_yscale("log")

'''

fig_conv, ax_conv = plt.subplots()

ax_conv.plot(std_in_sweep,conv_rate_sweep,label="Measured convergence rate (log 10 base)")
ax_conv.plot(std_in_sweep,conv_rate_approx_sweep,label="Approximated convergence rate (log 10 base)")
ax_conv.set_xlabel(r'$\sigma_e$')
ax_conv.legend()
plt.show()
