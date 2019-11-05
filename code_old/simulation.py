#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import os
import sys
from scipy.sparse import csr_matrix

import pdb

class driven_net:

    def __init__(self,
                 N_net,
                 cf_net,
                 cf_in,
                 std_conn,
                 std_in,
                 mu_act_target,
                 std_act_target,
                 eps_bias,
                 eps_gain,
                 eps_trail_av_error,
                 eps_trail_av_act,
                 n_t,
                 t_ext_off,
                 W_gen = np.random.normal,
                 **kwargs):

        self.N_net = N_net
        self.cf_net = cf_net
        self.cf_in = cf_in
        self.std_conn = std_conn
        self.std_in = std_in
        self.mu_act_target = mu_act_target
        self.std_act_target = std_act_target
        self.eps_bias = eps_bias
        self.eps_gain = eps_gain
        self.eps_trail_av_error = eps_trail_av_error
        self.eps_trail_av_act = eps_trail_av_act
        self.n_t = n_t
        self.t_ext_off = t_ext_off

        # Recording
        self.rec_options = {
            'x_rec': True,
            'x_trail_av_rec': True,
            'I_in_rec': True,
            'bias_rec': True,
            'gain_rec': True,
            'var_mean_rec': True}

        # Init state vars
        self.x_net = np.random.normal(0., 1., (self.N_net))
        self.x_net_trail_av = np.ones((self.N_net))*self.mu_act_target

        # Init Weights
        self.W = W_gen(0., std_conn, (N_net, N_net))
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
        else:
            self.x_net_rec = None

        if self.rec_options['x_trail_av_rec']:
            self.x_net_trail_av_rec = np.ndarray((n_t, N_net))
        else:
            self.x_net_trail_av_rec = None

        if self.rec_options['I_in_rec']:
            self.I_in_rec = np.ndarray((n_t, N_net))
        else:
            self.I_in_rec = None

        if self.rec_options['bias_rec']:
            self.bias_rec = np.ndarray((n_t, N_net))
        else:
            self.bias_rec = None

        if self.rec_options['gain_rec']:
            self.gain_rec = np.ndarray((n_t, N_net))
        else:
            self.gain_rec = None

        if self.rec_options['var_mean_rec']:
            self.var_mean_rec = np.ndarray((n_t))
        else:
            self.var_mean_rec = None

        #x_in = np.zeros((N_in))

        self.gain = np.ones((self.N_net))#np.random.rand(self.N_net)
        self.bias = np.random.normal(0.,.1,(self.N_net))

    def s(self, x):
        return np.tanh(x)

    def run_sim(self):

        self.W = csr_matrix(self.W)

        np.random.seed(self.rand_inp_seed)

        # Main Loop
        for t in tqdm(range(self.n_t)):

            if t < self.t_ext_off:
                #x_in = np.random.rand(N_in)
                I_in = np.random.normal(0., self.std_in, (self.N_net))
            else:
                #x_in = np.zeros((N_in))
                I_in = np.zeros((self.N_net))


            I = self.gain * (self.W.dot(self.x_net) + I_in - self.bias)

            if t < self.t_ext_off:
                self.gain += self.eps_gain * \
                    (self.std_act_target**2 - (self.x_net - self.x_net_trail_av)**2)
                self.bias += self.eps_bias * (self.x_net - self.mu_act_target)

            self.trail_av_hom_error += self.eps_trail_av_error * \
                (-self.trail_av_hom_error + ((self.std_act_target **
                                              2 - self.x_net**2)**2).sum()**.5 / self.N_net)



            self.x_net = self.s(I)

            self.x_net_trail_av += self.eps_trail_av_act*(self.x_net - self.x_net_trail_av)


            if self.rec_options['x_rec']:
                self.x_net_rec[t, :] = self.x_net

            if self.rec_options['x_trail_av_rec']:
                self.x_net_trail_av_rec[t, :] = self.x_net_trail_av

            if self.rec_options['I_in_rec']:
                self.I_in_rec[t, :] = I_in

            if self.rec_options['bias_rec']:
                self.bias_rec[t, :] = self.bias

            if self.rec_options['gain_rec']:
                self.gain_rec[t, :] = self.gain

            if self.rec_options['var_mean_rec']:
                self.var_mean_rec[t] = (self.x_net**2).mean()

        self.W = np.array(self.W.todense())

    def get_params(self):
        dict = {"N":self.N_net,
        "cf":self.cf_net,
        "std_conn":self.std_conn,
        "std_in":self.std_in,
        "mu_act_target":self.mu_act_target,
        "std_act_target":self.std_act_target,
        "eps_bias":self.eps_bias,
        "eps_gain":self.eps_gain,
        "n_t":self.n_t,
        "t_ext_off":self.t_ext_off,
        "rec_options":self.rec_options}

        return dict

    def save_data(self, folder, filename="sim_results.npz"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(folder + filename,
                            N_net=self.N_net,
                            cf_net=self.cf_net,
                            std_conn=self.std_conn,
                            std_in=self.std_in,
                            mu_act_target=self.mu_act_target,
                            std_act_target=self.std_act_target,
                            eps_bias=self.eps_bias,
                            eps_gain=self.eps_gain,
                            n_t=self.n_t,
                            t_ext_off=self.t_ext_off,
                            x_net_rec=self.x_net_rec,
                            x_net_trail_av_rec=self.x_net_trail_av_rec,
                            I_in_rec=self.I_in_rec,
                            bias_rec=self.bias_rec,
                            gain_rec=self.gain_rec,
                            trail_av_hom_error=self.trail_av_hom_error,
                            var_mean_rec=self.var_mean_rec,
                            W=self.W)


class driven_net_simple:

    def __init__(self,
                 N_net=1000,
                 cf_net=0.1,
                 std_conn=1.,
                 std_in=0.1,
                 gain=1.,
                 eps_trail_av_act=0.001,
                 n_t=100000,
                 t_ext_off=100000,
                 W_gen = np.random.normal,
                 **kwargs):

        self.N_net = N_net
        self.cf_net = cf_net
        self.std_conn = std_conn
        self.std_in = std_in
        self.eps_trail_av_act = eps_trail_av_act
        self.n_t = n_t
        self.t_ext_off = t_ext_off

        # Recording
        self.rec_options = {
            'x_rec': True,
            'x_trail_av_rec': True,
            'I_in_rec': True,
            'var_mean_rec': True}

        # Init state vars
        self.x_net = np.random.normal(0., 1., (self.N_net))
        self.x_net_trail_av = np.zeros((self.N_net))

        # Init Weights
        self.W = W_gen(0., std_conn, (N_net, N_net))
        self.W *= (np.random.rand(N_net, N_net) <=
                   cf_net) / (N_net * cf_net)**.5
        self.W[range(N_net), range(N_net)] = 0.

        #self.W = csr_matrix(self.W)

        # Init random seed:
        self.rand_inp_seed = np.random.randint(1000000)

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
        else:
            self.x_net_rec = None

        if self.rec_options['x_trail_av_rec']:
            self.x_net_trail_av_rec = np.ndarray((n_t, N_net))
        else:
            self.x_net_trail_av_rec = None

        if self.rec_options['I_in_rec']:
            self.I_in_rec = np.ndarray((n_t, N_net))
        else:
            self.I_in_rec = None

        if self.rec_options['var_mean_rec']:
            self.var_mean_rec = np.ndarray((n_t))
        else:
            self.var_mean_rec = None

        #x_in = np.zeros((N_in))

        self.gain = np.ones((self.N_net))*gain#np.random.rand(self.N_net)
        self.bias = np.zeros((self.N_net))

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


            I = self.gain * (self.W.dot(self.x_net) + I_in - self.bias)

            self.x_net = self.s(I)

            self.x_net_trail_av += self.eps_trail_av_act*(self.x_net - self.x_net_trail_av)


            if self.rec_options['x_rec']:
                self.x_net_rec[t, :] = self.x_net

            if self.rec_options['x_trail_av_rec']:
                self.x_net_trail_av_rec[t, :] = self.x_net_trail_av

            if self.rec_options['I_in_rec']:
                self.I_in_rec[t, :] = I_in

            if self.rec_options['var_mean_rec']:
                self.var_mean_rec[t] = (self.x_net**2).mean()

    def get_params(self):
        dict = {"N":self.N_net,
        "cf":self.cf_net,
        "std_conn":self.std_conn,
        "std_in":self.std_in,
        "n_t":self.n_t,
        "t_ext_off":self.t_ext_off,
        "rec_options":self.rec_options}

        return dict

    def save_data(self, folder, filename="sim_results.npz"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez_compressed(folder + filename,
                            N_net=self.N_net,
                            cf_net=self.cf_net,
                            std_conn=self.std_conn,
                            std_in=self.std_in,
                            n_t=self.n_t,
                            t_ext_off=self.t_ext_off,
                            x_net_rec=self.x_net_rec,
                            x_net_trail_av_rec=self.x_net_trail_av_rec,
                            I_in_rec=self.I_in_rec,
                            var_mean_rec=self.var_mean_rec,
                            W=self.W)
