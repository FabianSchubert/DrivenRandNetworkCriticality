#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import os
import sys


class driven_net:

    def __init__(self,
                 N_net,
                 cf_net,
                 std_conn,
                 std_in,
                 std_act_target,
                 mu_gain,
                 mu_trail_av_error,
                 n_t,
                 t_ext_off,
                 **kwargs):

        self.N_net = N_net
        self.cf_net = cf_net
        self.std_conn = std_conn
        self.std_in = std_in
        self.std_act_target = std_act_target
        self.mu_gain = mu_gain
        self.mu_trail_av_error = mu_trail_av_error
        self.n_t = n_t
        self.t_ext_off = t_ext_off

        # Recording
        self.rec_options = {
            'x_rec': True,
            'I_in_rec': True,
            'gain_rec': True,
            'var_mean_rec': True}

        # Init state vars
        self.x_net = np.random.normal(0., 1., (self.N_net))

        # Init Weights
        self.W = np.random.normal(0., std_conn, (N_net, N_net))
        self.W *= (np.random.rand(N_net, N_net) <=
                   cf_net) / (N_net * cf_net)**.5
        self.W[range(N_net), range(N_net)] = 0.

        # Init random seed:
        self.rand_inp_seed = np.random.randint(1000000)

        self.trail_av_hom_error = 0.

        if kwargs is not None:
            for key, value in kwargs.items():
                if key in self.rec_options.keys():
                    self.rec_options[key] = value
                elif key == "x_init":
                    self.x_net = value
                elif key == "W_init":
                    self.W = value
                elif key == "rand_inp_seed":
                    self.rand_inp_seed = value
                else:
                    print("Invalid keyword argument!")
                    sys.exit()

        if self.rec_options['x_rec']:
            self.x_net_rec = np.ndarray((n_t, N_net))

        if self.rec_options['I_in_rec']:
            self.I_in_rec = np.ndarray((n_t, N_net))

        if self.rec_options['gain_rec']:
            self.gain_rec = np.ndarray((n_t, N_net))

        if self.rec_options['var_mean_rec']:
            self.var_mean_rec = np.ndarray((n_t))

        #x_in = np.zeros((N_in))

        self.gain = np.ones((self.N_net))

    def s(self, x):
        return np.tanh(x)

    def run_sim(self):

        np.random.seed(self.rand_inp_seed)

        # Main Loop
        for t in tqdm(range(self.n_t)):

            if t < self.t_ext_off:
                #x_in = np.random.rand(N_in)
                I_in = np.random.normal(0., self.std_in, (self.N_net))
            else:
                #x_in = np.zeros((N_in))
                I_in = np.zeros((self.N_net))

            I = self.gain * (np.dot(self.W, self.x_net) + I_in)

            if t < self.t_ext_off:
                self.gain += self.mu_gain * \
                    (self.std_act_target**2 - self.x_net**2)

            self.trail_av_hom_error += self.mu_trail_av_error * \
                (-self.trail_av_hom_error + ((self.std_act_target **
                                              2 - self.x_net**2)**2).sum()**.5 / self.N_net)

            self.x_net = self.s(I)

            if self.rec_options['x_rec']:
                self.x_net_rec[t, :] = self.x_net

            if self.rec_options['I_in_rec']:
                self.I_in_rec[t, :] = I_in

            if self.rec_options['gain_rec']:
                self.gain_rec[t, :] = self.gain

            if self.rec_options['var_mean_rec']:
                self.var_mean_rec[t] = (self.x_net**2).mean()

    def save_data(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(folder + "sim_results.npz",
                            N_net=self.N_net,
                            cf_net=self.cf_net,
                            std_conn=self.std_conn,
                            std_in=self.std_in,
                            std_act_target=self.std_act_target,
                            mu_gain=self.mu_gain,
                            n_t=self.n_t,
                            t_ext_off=self.t_ext_off,
                            x_net_rec=self.x_net_rec,
                            I_in_rec=self.I_in_rec,
                            gain_rec=self.gain_rec,
                            trail_av_hom_error=self.trail_av_hom_error,
                            var_mean_rec=self.var_mean_rec,
                            W=self.W)
