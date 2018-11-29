#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import os

class driven_net:

    def __init__(self,
                N_net,
                cf_net,
                std_conn,
                std_in,
                var_act_target,
                mu_gain,
                n_t,
                t_ext_off):

        self.N_net = N_net
        self.cf_net = cf_net
        self.std_conn = std_conn
        self.std_in = std_in
        self.var_act_target = var_act_target
        self.mu_gain = mu_gain
        self.n_t = n_t
        self.t_ext_off = t_ext_off

        ### Recording
        self.x_net_rec = np.ndarray((n_t, N_net))
        self.I_in_rec = np.ndarray((n_t, N_net))

        self.gain_rec = np.ndarray((n_t, N_net))

        self.var_mean_rec = np.ndarray((n_t))

        ### Init Weights
        self.W = np.random.normal(0., std_conn, (N_net,N_net))
        self.W *= (np.random.rand(N_net, N_net) <= cf_net) / ( N_net * cf_net )**.5
        self.W[range(N_net), range(N_net)] = 0.

    def s(self,x):
        return np.tanh(x)

    def run_sim(self):

        x_net = np.random.normal(0., 1., (self.N_net))
        #x_in = np.zeros((N_in))

        gain = np.ones((self.N_net))

        ### Main Loop
        for t in tqdm(range(self.n_t)):

            if t < self.t_ext_off:
                #x_in = np.random.rand(N_in)
                I_in = np.random.normal(0.,self.std_in,(self.N_net))
            else:
                #x_in = np.zeros((N_in))
                I_in = np.zeros((self.N_net))

            I = gain * ( np.dot(self.W, x_net) + I_in )

            if t < self.t_ext_off:
                gain += self.mu_gain * ( self.var_act_target - x_net**2 )

            x_net = self.s(I)

            self.x_net_rec[t,:] = x_net
            self.I_in_rec[t,:] = I_in
            self.gain_rec[t,:] = gain
            self.var_mean_rec[t] = (x_net**2).mean()
    def save_data(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(folder + "sim_results.npz",
        N_net = self.N_net
        cf_net = self.cf_net = cf_net
        std_conn = self.std_conn = std_conn
        std_in = self.std_in = std_in
        var_act_target = self.var_act_target = var_act_target
        mu_gain = self.mu_gain = mu_gain
        n_t = self.n_t = n_t
        t_ext_off = self.t_ext_off = t_ext_off
        x_net_rec = self.x_net_rec
        I_in_rec = self.I_in_rec = np.ndarray((n_t, N_net))
        gain_rec = self.gain_rec = np.ndarray((n_t, N_net))
        var_mean_rec = self.var_mean_rec = np.ndarray((n_t))
        W = self.W = np.random.normal(0., std_conn, (N_net,N_net))  )
