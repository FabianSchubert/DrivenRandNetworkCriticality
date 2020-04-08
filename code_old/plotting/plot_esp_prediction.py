#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
plt.style.use('matplotlibrc')

import matplotlib as mpl

from plot_std_in_std_target_sweep import plot_echo_state_prop_trans as plot_esp_trans

from scipy import integrate
from scipy.optimize import root

def int_func(I,a,sigm_y,sigm_e,sigm_w):
    return np.tanh(a*I)**2.*np.exp(-I**2./(2.*(sigm_y**2.*sigm_w**2.+sigm_e**2.)))/(np.pi*2.*(sigm_y**2.*sigm_w**2.+sigm_e**2.))**.5

def integral(a,sigm_y,sigm_e,sigm_w):
    return integrate.quad(lambda x: int_func(x,a,sigm_y,sigm_e,sigm_w), -np.inf,np.inf)[0]

def root_func(a,sigm_y,sigm_e,sigm_w):
    return integral(a,sigm_y,sigm_e,sigm_w) - sigm_y**2.

def int_func_deriv(I,a,sigm_y,sigm_e,sigm_w):
    return (1.-np.tanh(a*I)**2.)**2.*np.exp(-I**2./(2.*(sigm_y**2.*sigm_w**2.+sigm_e**2.)))/(np.pi*2.*(sigm_y**2.*sigm_w**2.+sigm_e**2.))**.5

def integral_deriv(a,sigm_y,sigm_e,sigm_w):
    return integrate.quad(lambda x: int_func_deriv(x,a,sigm_y,sigm_e,sigm_w), -10.,10.)[0]

def a(s_t,s_e,s_w):
    return ((1.-(1.-s_t**2.)**2.)/(2.*(1.-s_t**2)**2.*(s_t**2.+s_e**2./s_w**2.)))**.5/s_w

def s_total(s_t,s_e,s_w):
    return a(s_t,s_e,s_w)*s_w*(s_t**2.+s_e**2./s_w**2.)**.5

def specrad(s_t,s_e,s_w):
    return a(s_t,s_e,s_w)*s_w*(1.+4.*s_total(s_t,s_e,s_w)**2.)**-.25

def a_full(s_t,s_e,s_w):
    if type(s_t) == np.ndarray and type(s_e) == np.ndarray:
        a_out = np.ndarray(s_t.shape)
        for k in range(s_t.shape[0]):
            for l in range(s_t.shape[1]):
                result_a = root(lambda a: root_func(a,s_t[k,l],s_e[k,l],s_w),a(s_t[k,l],s_e[k,l],s_w))
                if result_a['success']:
                    a_out[k,l] = result_a['x'][0]
                else:
                    a_out[k,l] = np.nan
        return a_out
    else:
        result_a = root(lambda a: root_func(a,s_t,s_e,s_w),a(s_t,s_e,s_w))
        if result_a['success']:
            return result_a['x'][0]
        else:
            return np.nan

def scale_full(s_t,s_e,s_w):
    if type(s_t) == np.ndarray and type(s_e) == np.ndarray:
        scale_out = np.ndarray(s_t.shape)
        a_arr = a_full(s_t,s_e,s_w)
        for k in range(s_t.shape[0]):
            for l in range(s_t.shape[1]):
                scale_out[k,l] = a_arr[k,l] * s_w * integral_deriv(a_arr[k,l],s_t[k,l],s_e[k,l],s_w)**.5
        return scale_out
    else:
        af = a_full(s_t,s_e,s_w)
        return af * s_w * integral_deriv(af,s_t,s_e,s_w)**.5

Data0p5 = "../../data/max_lyap_sweep/sim_results_mc_sigmaw_0p5_1.npz"
Data1p0 = "../../data/max_lyap_sweep/sim_results_mc_sigmaw_1p0_1.npz"
Data2p0 = "../../data/max_lyap_sweep/sim_results_mc_sigmaw_2p0_1.npz"

s_t = np.load(Data1p0)["std_act_target_sweep_range"]
s_e = np.load(Data1p0)["std_in_sweep_range"]

S_T,S_E = np.meshgrid(s_t,s_e)


textwidth = 5.5532

figure_file = "../../plots/esp_prediction"

output_formats = ['pdf','png']

fig, ax = plt.subplots(figsize=(textwidth,0.6*textwidth))

cmap = mpl.cm.get_cmap('viridis')

s_t_highres = np.linspace(s_t[0],s_t[-1],60)
s_e_highres = np.linspace(s_e[0],s_e[-1],60)

S_T_HR,S_E_HR = np.meshgrid(s_t_highres,s_e_highres)

sc_full_0p5 = scale_full(S_T_HR,S_E_HR,.5)
sc_full_1p0 = scale_full(S_T_HR,S_E_HR,1.)
sc_full_2p0 = scale_full(S_T_HR,S_E_HR,2.)

plot_esp_trans(ax,color=cmap(0.),file_path=Data0p5,sigmw=.5)
plot_esp_trans(ax,color=cmap(.5),file_path=Data1p0,sigmw=1.)
plot_esp_trans(ax,color=cmap(1.),file_path=Data2p0,sigmw=2.)

ax.contour(s_t,s_e/.5,specrad(S_T,S_E,0.5),levels=[1.],colors=[cmap(0.)],linestyles='dashed')
ax.contour(s_t,s_e/1.,specrad(S_T,S_E,1.0),levels=[1.],colors=[cmap(0.5)],linestyles='dashed')
ax.contour(s_t,s_e/2.,specrad(S_T,S_E,2.0),levels=[1.],colors=[cmap(1.)],linestyles='dashed')

ax.contour(s_t_highres,s_e_highres/.5,sc_full_0p5,levels=[1.],colors=[cmap(0.)],linewidths=[2.],zorder=10)
ax.contour(s_t_highres,s_e_highres/1.,sc_full_1p0,levels=[1.],colors=[cmap(0.5)],linewidths=[2.],zorder=10)
ax.contour(s_t_highres,s_e_highres/2.,sc_full_2p0,levels=[1.],colors=[cmap(1.)],linewidths=[2.],zorder=10)

ax.set_ylim([0.,s_e[-1]*0.8])

ax.legend()

fig.tight_layout(pad=0.1)

for format in output_formats:
    if format=="png":
        fig.savefig(figure_file+"."+format, dpi=1000)
    else:
        fig.savefig(figure_file+"."+format)

plt.show()
