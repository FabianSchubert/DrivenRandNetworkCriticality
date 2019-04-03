#!/usr/bin/env python3

import numpy as np

import pdb

class esn:

    def __init__(self,N=1000,cf=1.,spec_rad=0.9,sigm_w_in=0.5,cf_w_in=0.1,data_dim=1,reg_fact=0.01):

        self.N = N
        self.spec_rad = spec_rad

        self.W = np.random.normal(0.,spec_rad/(N*cf)**.5,(N,N))*(np.random.rand(N,N) <= cf)
        self.W[range(N),range(N)] = 0.

        self.data_dim = data_dim

        self.sigm_w_in = sigm_w_in
        self.w_in = np.random.normal(0.,sigm_w_in*cf_w_in,(N,data_dim))*(np.random.rand(N,data_dim) <= cf_w_in)

        self.w_out = np.ndarray((data_dim,N+1))

        self.reg_fact = reg_fact


    def learn_w_out(self,u_in,u_target):

        n_t = u_in.shape[0]

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = np.tanh(self.w_in @ u_in[0,:])

        for t in range(1,n_t):

            y[t,1:] = np.tanh(self.W @ y[t-1,1:] + self.w_in @ u_in[t,:])

        self.w_out[:,:] = (np.linalg.inv(y.T @ y + self.reg_fact*np.eye(self.N+1)) @ y.T @ u_target).T


    def predict_data(self,data):

        n_t = data.shape[0]

        u_in = data

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = np.tanh(self.w_in @ u_in[0,:])

        for t in range(1,n_t):

            y[t,1:] = np.tanh(self.W @ y[t-1,1:] + self.w_in @ u_in[t,:])

        return (self.w_out @ y.T).T
