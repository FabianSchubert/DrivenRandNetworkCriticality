#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

def s(x):
    return np.tanh(x)

### Parameters
N_net = 500
N_in = 10

cf_net = 0.1
cf_in = 0.1

std_conn = 1.

var_act_target = 0.1**2

mu_gain = 0.001

n_t = 50000

W_net = np.random.normal(0., std_conn, (N_net,N_net))
W_net *= (np.random.rand(N_net, N_net) <= cf_net) / ( N_net * cf_net )**.5

W_in = np.random.normal(0., std_conn, (N_net,N_in))
W_in *= (np.random.rand(N_net, N_in) <= cf_in) / ( N_in * cf_in )**.5

x_net = np.random.normal(0., 1., (N_net))
x_in = np.zeros((N_in))

gain = np.ones((N_net))
###

### Recording
x_net_rec = np.ndarray((n_t, N_net))
x_in_rec = np.ndarray((n_t, N_in))

gain_rec = np.ndarray((n_t, N_net))

var_mean_rec = np.ndarray((n_t))
###

### Main Loop
for t in tqdm(range(n_t)):

    I = gain * np.dot(W_net, x_net)

    gain += mu_gain * ( var_act_target - x_net**2 )

    x_net = s(I)

    x_net_rec[t,:] = x_net
    gain_rec[t,:] = gain
    var_mean_rec[t] = (x_net**2).mean()
###

### Plotting
fig_act, ax_act = plt.subplots()
ax_act.plot(x_net_rec[-100:,:5])

fig_gain, ax_gain = plt.subplots()
ax_gain.plot(gain_rec[:,::10])

fig_var, ax_var = plt.subplots()
ax_var.plot(var_mean_rec)

plt.show()
###

pdb.set_trace()
