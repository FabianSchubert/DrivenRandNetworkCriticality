#!/usr/bin/env python3

import numpy as np

import pdb

import sys


def in_ipynb():
    try:
        cfg = get_ipython().__class__.__name__
        return True
    except NameError:
        return False

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

class esn:

    def __init__(self,
        N=1000,
        cf=1.,
        sigm_w=1.,
        sigm_w_in=1.,
        alpha=1.,
        cf_w_in=0.1,
        data_dim_in=1,
        data_dim_out=1,
        reg_fact=0.01,
        bias=0.,
        gain=1.,
        eps_gain=0.001,
        eps_bias=0.0002,
        mu_act_target=0.,
        sigm_act_target=0.25,
        eps_trail_av_act=0.0001,
        eps_trail_av_mu_ext=0.0001,
        eps_trail_av_sigm_ext=0.001,
        W_gen=np.random.normal,
        w_in_gen=np.random.normal):

        self.N = N

        self.sigm_w = sigm_w

        self.W = W_gen(0.,sigm_w/(N*cf)**.5,(N,N))*(np.random.rand(N,N) <= cf)
        self.W[range(N),range(N)] = 0.

        self.data_dim_in = data_dim_in
        self.data_dim_out = data_dim_out

        self.sigm_w_in = sigm_w_in
        self.w_in = w_in_gen(0.,sigm_w_in,(N,data_dim_in))*(np.random.rand(N,data_dim_in) <= cf_w_in)#np.random.normal(0.,sigm_w_in,(N,data_dim_in))*(np.random.rand(N,data_dim_in) <= cf_w_in)

        self.w_out = np.ndarray((data_dim_out,N+1))

        self.reg_fact = reg_fact

        self.cf = cf
        self.cf_w_in = cf_w_in

        self.alpha = alpha

        self.bias = bias * np.ones((self.N))
        self.gain = gain * np.ones((self.N))

        self.eps_gain = eps_gain
        self.eps_bias = eps_bias

        self.eps_trail_av_act = eps_trail_av_act

        self.eps_trail_av_mu_ext=eps_trail_av_mu_ext
        self.eps_trail_av_sigm_ext=eps_trail_av_sigm_ext

        self.mu_act_target = mu_act_target
        self.sigm_act_target = sigm_act_target


    def check_data_in_comp(self,data):

        if len(data.shape)==1:
            if self.data_dim_in != 1:
                print("input dimensions do not fit!")
                sys.exit()
            return np.array([data]).T

        elif (len(data.shape)>2) or (data.shape[1] != self.data_dim_in):
            print("input dimensions do not fit!")
            sys.exit()

        return data

    def check_data_out_comp(self,data):

        if len(data.shape)==1:
            if self.data_dim_out != 1:
                print("output dimensions do not fit!")
                sys.exit()
            return np.array([data]).T

        elif (len(data.shape)>2) or (data.shape[1] != self.data_dim_out):
            print("output dimensions do not fit!")
            sys.exit()

        return data


    def learn_w_out(self,u_in,u_target,t_prerun=0,show_progress=True):

        u_in = self.check_data_in_comp(u_in)
        u_target = self.check_data_out_comp(u_target)

        n_t = u_in.shape[0]

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.


        y[0,1:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))


        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y[t,1:] = y[t-1,1:]*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y[t-1,1:] + self.w_in @ u_in[t,:] - self.bias))

        self.w_out[:,:] = (np.linalg.inv(y[t_prerun:,:].T @ y[t_prerun:,:] + self.reg_fact*np.eye(self.N+1)) @ y[t_prerun:,:].T @ u_target[t_prerun:,:]).T


    def predict_data(self,data,return_reservoir_rec=False,show_progress=True):

        data = self.check_data_in_comp(data)

        n_t = data.shape[0]

        u_in = data

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y[t,1:] = y[t-1,1:]*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y[t-1,1:] + self.w_in @ u_in[t,:] - self.bias))

        out = (self.w_out @ y.T).T
        if self.data_dim_out == 1:
            out = out[:,0]

        if return_reservoir_rec:
            return (out,y)
        else:
            return out

    def W_gain(self):
        return (self.W.T*self.gain).T

    def run_hom_adapt(self,u_in,return_reservoir_rec=False,return_gain_rec=False,return_bias_rec=False,show_progress=True,subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        if return_reservoir_rec:
            y_rec = np.ndarray((n_rec,self.N))
            y_rec[0,:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y = y_rec[0,:]
            y_trail_av = y_rec[0,:]
        else:
            y = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y_trail_av = np.array(y)

        if return_gain_rec:
            gain_rec = np.ndarray((n_rec,self.N))
            gain_rec[0,:] = self.gain

        if return_bias_rec:
            bias_rec = np.ndarray((n_rec,self.N))
            bias_rec[0,:] = self.bias

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)
            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + self.w_in @ u_in[t,:] - self.bias))

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)

            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)
                if return_reservoir_rec:
                    y_rec[t_rec,:] = y
                if return_bias_rec:
                    bias_rec[t_rec,:] = self.bias
                if return_gain_rec:
                    gain_rec[t_rec,:] = self.gain



        returndat = []

        if return_reservoir_rec:
            returndat.append(y_rec)
        if return_bias_rec:
            returndat.append(bias_rec)
        if return_gain_rec:
            returndat.append(gain_rec)

        if len(returndat)==1:
            return returndat[0]
        elif len(returndat)>1:
            return returndat

    def run_hom_adapt_auto_target(self,
                                u_in,
                                return_reservoir_rec=False,
                                return_gain_rec=False,
                                return_bias_rec=False,
                                return_mu_ext=False,
                                return_sigm_ext=False,
                                show_progress=True,
                                subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        if return_reservoir_rec:
            y_rec = np.ndarray((n_rec,self.N))
            y_rec[0,:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y = y_rec[0,:]
            y_trail_av = y_rec[0,:]
        else:
            y = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y_trail_av = np.array(y)

        if return_gain_rec:
            gain_rec = np.ndarray((n_rec,self.N))
            gain_rec[0,:] = self.gain

        if return_bias_rec:
            bias_rec = np.ndarray((n_rec,self.N))
            bias_rec[0,:] = self.bias

        mu_ext = self.w_in @ u_in[0,:]
        if return_mu_ext:
            mu_ext_rec = np.ndarray((n_rec,self.N))
            mu_ext_rec[0,:] = mu_ext

        sigm_ext_squ = np.ones((self.N))*10.**-3.
        if return_sigm_ext:
            sigm_ext_rec = np.ndarray((n_rec,self.N))
            sigm_ext_rec[0,:] = sigm_ext_squ**.5

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            ext_in = self.w_in @ u_in[t,:]

            mu_ext = self.eps_trail_av_mu_ext*ext_in + (1.-self.eps_trail_av_mu_ext)*mu_ext

            sigm_ext_squ = self.eps_trail_av_sigm_ext*(ext_in - mu_ext)**2. + (1.-self.eps_trail_av_sigm_ext)*sigm_ext_squ

            sigm_ext = sigm_ext_squ**.5

            self.sigm_act_target = ((1.5)**.5 * self.sigm_w/sigm_ext + 1.)**(-.5)

            self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)
            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + ext_in - self.bias))

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)

            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)
                if return_reservoir_rec:
                    y_rec[t_rec,:] = y
                if return_bias_rec:
                    bias_rec[t_rec,:] = self.bias
                if return_gain_rec:
                    gain_rec[t_rec,:] = self.gain
                if return_mu_ext:
                    mu_ext_rec[t_rec,:] = mu_ext
                if return_sigm_ext:
                    sigm_ext_rec[t_rec,:] = sigm_ext

        returndat = []

        if return_reservoir_rec:
            returndat.append(y_rec)
        if return_bias_rec:
            returndat.append(bias_rec)
        if return_gain_rec:
            returndat.append(gain_rec)
        if return_mu_ext:
            returndat.append(mu_ext_rec)
        if return_sigm_ext:
            returndat.append(sigm_ext_rec)

        if len(returndat)==1:
            return returndat[0]
        elif len(returndat)>1:
            return returndat

    def run_hom_adapt_auto(self,
                            u_in,
                            return_reservoir_rec=False,
                            return_gain_rec=False,
                            return_bias_rec=False,
                            return_mu_ext=False,
                            return_sigm_ext=False,
                            show_progress=True,
                            subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        if return_reservoir_rec:
            y_rec = np.ndarray((n_rec,self.N))
            y_rec[0,:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y = y_rec[0,:]
            y_trail_av = y_rec[0,:]
        else:
            y = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y_trail_av = np.array(y)

        if return_gain_rec:
            gain_rec = np.ndarray((n_rec,self.N))
            gain_rec[0,:] = self.gain

        if return_bias_rec:
            bias_rec = np.ndarray((n_rec,self.N))
            bias_rec[0,:] = self.bias

        mu_ext = self.w_in @ u_in[0,:]
        if return_mu_ext:
            mu_ext_rec = np.ndarray((n_rec,self.N))
            mu_ext_rec[0,:] = mu_ext

        sigm_ext_squ = np.ones((self.N))*10.**-3.
        if return_sigm_ext:
            sigm_ext_rec = np.ndarray((n_rec,self.N))
            sigm_ext_rec[0,:] = sigm_ext_squ**.5

        mu_recurr = np.zeros((self.N))
        if return_mu_recurr:
            mu_recurr_rec = np.ndarray((n_rec,self.N))
            mu_recurr_rec[0,:] = mu_recurr

        sigm_recurr_squ = np.ones((self.N))*10.**-3.
        if return_sigm_recurr:
            sigm_recurr_rec = np.ndarray((n_rec,self.N))
            sigm_recurr_rec[0,:] = sigm_recurr_squ**.5

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            ext_in = self.w_in @ u_in[t,:]

            mu_ext = self.eps_trail_av_mu_ext*ext_in + (1.-self.eps_trail_av_mu_ext)*mu_ext

            sigm_ext_squ = self.eps_trail_av_sigm_ext*(ext_in - mu_ext)**2. + (1.-self.eps_trail_av_sigm_ext)*sigm_ext_squ

            sigm_ext = sigm_ext_squ**.5

            self.sigm_act_target = ((1.5)**.5 * self.sigm_w/sigm_ext + 1.)**(-.5)

            self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)
            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + ext_in - self.bias))

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)

            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)
                if return_reservoir_rec:
                    y_rec[t_rec,:] = y
                if return_bias_rec:
                    bias_rec[t_rec,:] = self.bias
                if return_gain_rec:
                    gain_rec[t_rec,:] = self.gain
                if return_mu_ext:
                    mu_ext_rec[t_rec,:] = mu_ext
                if return_sigm_ext:
                    sigm_ext_rec[t_rec,:] = sigm_ext

        returndat = []

        if return_reservoir_rec:
            returndat.append(y_rec)
        if return_bias_rec:
            returndat.append(bias_rec)
        if return_gain_rec:
            returndat.append(gain_rec)
        if return_mu_ext:
            returndat.append(mu_ext_rec)
        if return_sigm_ext:
            returndat.append(sigm_ext_rec)

        if len(returndat)==1:
            return returndat[0]
        elif len(returndat)>1:
            return returndat
