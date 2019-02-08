#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

def input_gen(t):

    return np.random.rand()-0.5

def reward(y_train,y_out):
    return 2.-np.abs(y_train-y_out)

n = 1500

n_learn_samples = 5000
t_back = 10

mu_w_recurr = 0.0001
mu_w_out = 0.0005

W = np.random.normal(0.,1./n**.5,(n,n))
W[range(n),range(n)] = 0.

W_copy = np.array(W)

W_rand_back = np.random.rand(n,n)
W_rand_back[range(n),range(n)] = 0.

w_in = np.random.normal(0.,1.,(n))

w_out = np.random.normal(0.,1./n,(n+1))

# input signal u
u = np.ndarray((n_learn_samples+t_back))
# echo state
x = np.ndarray((n_learn_samples,n+1))# use n+1 to add a bias
# target readout
y = np.ndarray((n_learn_samples,1))
y_out = np.ndarray((n_learn_samples,1))

t = 0

u[t] = input_gen(t)
x_prerun = np.tanh(u[t]*w_in)

x[:,0] = 1. #for bias

for t in range(1,t_back):
    u[t] = input_gen(t)
    x_prerun = np.tanh(np.dot(W,x_prerun)+u[t]*w_in)

for t in tqdm(range(n_learn_samples)):
    u[t+t_back] = input_gen(t+t_back)
    y[t] = u[t]
    if t == 0:
        x[t,1:] = np.tanh(np.dot(W,x_prerun) + u[t+t_back]*w_in)
    else:
        x[t,1:] = np.tanh(np.dot(W,x[t-1,1:]) + u[t+t_back]*w_in)

    y_out[t] = np.dot(x[t,:],w_out)

    w_out += mu_w_out * x[t,:] * (y[t] - y_out[t])



    W += mu_w_recurr * np.outer(x[t,1:],x[t-1,1:]) * reward(y[t],y_out[t])# * W_rand_back
    W[range(n),range(n)] = 0.
    W_std = W.std(axis=1)
    W = (W.T/(W_std*n**.5)).T

    #W[range(n),range(n)] = 0.


fig, ax = plt.subplots(1,1)
ax.plot(y_out)
ax.plot(y)

fig2, ax2 = plt.subplots(1,1)
ax2.plot((y_out-y)**2)

fig3, ax3 = plt.subplots(1,1)
ax3.plot(1./(1.+5.*(y_out-y)**2))

plt.show()

pdb.set_trace()
################# ridge regression
#w_out_est = np.linalg.inv(x.T @ x + reg_fact*np.eye(n+1)) @ x.T @ y
#w_out_est = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x) + reg_fact*np.eye(n+1)),x.T),y)

#y_est = x @ w_out_est
