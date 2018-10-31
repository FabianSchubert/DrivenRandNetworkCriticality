#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def s(x):
    return np.tanh(x)

N_net = 300
N_in = 10

cf_net = 0.1
cf_in = 0.1

std_conn = 1.

W_net = np.random.normal(0., std_conn, (N_net,N_net))
W_net *= (np.random.rand(N_net, N_net) <= cf_net)

W_in = np.random.normal(0., std_conn, (N_net,N_in))
W_net *= (np.random.rand(N_net, N_in) <= cf_in)

x_net = np.random.normal(0., 1., (N_net))
x_in = np.zeros((N_in))
#x_in = np.random.normal(0., 1., (N_in))

n_t = 10000

### Recording
x_net_rec = np.ndarray((n_t, N_net))
x_in_rec = np.ndarray((n_t, N_in))
###

for t in tqdm(range(n_t)):

    x_net = s(np.dot(W_net, x_net))


    x_net_rec[t,:] = x_net


fig, ax = plt.subplots()

ax.plot(x_net_rec)

plt.show()
