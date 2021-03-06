#!/usr/bin/env python3

import numpy as np

import pdb

import sys

import pdb

def in_ipynb():
    try:
        cfg = get_ipython().__class__.__name__
        return True
    except NameError:
        return False
'''
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
'''

from tqdm.autonotebook import tqdm

from scipy.optimize import curve_fit

from scipy.sparse import csr_matrix

class esn:

    def __init__(self,
        N=1000,
        cf=.1,
        sigm_w=1.,
        sigm_w_in=1.,
        alpha=1.,
        cf_w_in=1.,
        data_dim_in=1,
        data_dim_out=1,
        reg_fact=0.01,
        bias=0.,
        gain=1.,
        eps_gain=0.005,
        eps_bias=0.001,
        mu_act_target=0.,
        sigm_act_target=0.25,
        eps_trail_av_act=0.0005,
        eps_trail_av_sigm_act=0.005,
        eps_trail_av_mu_ext=0.0005,
        eps_trail_av_sigm_ext=0.005,
        eps_trail_av_mu_recurr=0.0005,
        eps_trail_av_sigm_recurr=0.005,
        eps_sigm_act_target=0.005,
        eps_LMS_out=0.001,
        eps_LMS_gain=0.0001,
        W_gen=np.random.normal,
        w_in_gen=np.random.normal,
        noise_level=0.):

        self.N = N

        self.sigm_w = sigm_w



        self.W = W_gen(0.,sigm_w/(N*cf)**.5,(N,N))*(np.random.rand(N,N) <= cf)
        self.W[range(N),range(N)] = 0.

        if cf < .5:
            self.W = csr_matrix(self.W)
            self.sparse_W = True
        else:
            self.sparse_W = False

        self.data_dim_in = data_dim_in
        self.data_dim_out = data_dim_out

        self.sigm_w_in = sigm_w_in
        self.w_in = w_in_gen(0.,sigm_w_in,(N,data_dim_in))*(np.random.rand(N,data_dim_in) <= cf_w_in)#np.random.normal(0.,sigm_w_in,(N,data_dim_in))*(np.random.rand(N,data_dim_in) <= cf_w_in)

        self.w_out = np.random.rand(data_dim_out,N+1) - .5
        self.w_out[:,0] = 0.

        self.reg_fact = reg_fact

        self.cf = cf
        self.cf_w_in = cf_w_in

        self.alpha = alpha

        self.bias = bias * np.ones((self.N))
        self.gain = gain * np.ones((self.N))

        self.eps_gain = eps_gain
        self.eps_bias = eps_bias

        self.eps_trail_av_act = eps_trail_av_act
        self.eps_trail_av_sigm_act = eps_trail_av_sigm_act

        self.eps_trail_av_mu_ext = eps_trail_av_mu_ext
        self.eps_trail_av_sigm_ext = eps_trail_av_sigm_ext

        self.eps_trail_av_mu_recurr = eps_trail_av_mu_recurr
        self.eps_trail_av_sigm_recurr = eps_trail_av_sigm_recurr

        self.eps_sigm_act_target = eps_sigm_act_target

        self.mu_act_target = mu_act_target
        self.sigm_act_target = sigm_act_target

        self.eps_LMS_out = eps_LMS_out
        self.eps_LMS_gain = eps_LMS_gain

        self.noise_level = noise_level

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

    def rescale_eps(self,s):
        self.eps_gain = self.eps_gain*s
        self.eps_bias = self.eps_bias*s

        self.eps_trail_av_act = self.eps_trail_av_act*s
        self.eps_trail_av_sigm_act = self.eps_trail_av_sigm_act*s

        self.eps_trail_av_mu_ext = self.eps_trail_av_mu_ext*s
        self.eps_trail_av_sigm_ext = self.eps_trail_av_sigm_ext*s

        self.eps_trail_av_mu_recurr = self.eps_trail_av_mu_recurr*s
        self.eps_trail_av_sigm_recurr = self.eps_trail_av_sigm_recurr*s


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


    def learn_w_out_online_LMS(self,u_in,u_target,t_prerun=0,show_progress=True,return_w_out=False):

        u_in = self.check_data_in_comp(u_in)
        u_target = self.check_data_out_comp(u_target)

        n_t = u_in.shape[0]

        if return_w_out:
            w_out_rec = np.ndarray((n_t,self.data_dim_out,self.N+1))

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y[t,1:] = y[t-1,1:]*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y[t-1,1:] + self.w_in @ u_in[t,:] - self.bias))

            u_out = self.w_out @ y[t,:]

            self.w_out += self.eps_LMS_out * np.tensordot((u_target[t] - u_out),y[t,:],axes=0)

            if return_w_out:
                w_out_rec[t,:,:] = self.w_out

        if return_w_out:
            return w_out_rec

    def learn_w_out_gains_online_LMS(self,u_in,u_target,t_prerun=0,show_progress=True,return_w_out=False,return_gain=False,subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)
        u_target = self.check_data_out_comp(u_target)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        if return_w_out:
            w_out_rec = np.ndarray((n_rec,self.data_dim_out,self.N+1))

        if return_gain:
            gain_rec = np.ndarray((n_rec,self.N))

        y = np.ndarray((self.N+1))
        y[0] = 1.

        y[1:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            X = self.W.dot(y[1:]) + self.w_in @ u_in[t,:] - self.bias

            y[1:] = y[1:]*(1.-self.alpha) + self.alpha*np.tanh(self.gain*X)

            u_out = self.w_out @ y[:]

            if t > t_prerun:
                self.gain += self.eps_LMS_gain * np.tensordot((u_target[t] - u_out),self.w_out[:,1:],axes=1)*(1.-y[1:]**2.)*X
                self.gain = np.maximum(0.,self.gain)
                #self.gain += self.eps_LMS_gain * np.tensordot((u_target[t] - u_out),self.w_out[:,1:],axes=1)*X

                self.w_out += self.eps_LMS_out * np.tensordot((u_target[t] - u_out),y[:],axes=0)

            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)
                if return_gain:
                    gain_rec[t_rec,:] = self.gain

                if return_w_out:
                    w_out_rec[t_rec,:,:] = self.w_out

        returndat = []

        if return_gain:
            returndat.append(gain_rec)

        if return_w_out:
            returndat.append(w_out_rec)

        if len(returndat) == 1:
            return returndat[0]
        elif len(returndat) > 1:
            return returndat


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
        if self.sparse_W:
            return (np.array(self.W.todense()).T*self.gain).T
        else:
            return (self.W.T*self.gain).T

    def esp_fit_func(self,x,f0,a,x1):

        return (x<=x1)*(f0 + a*x) + (x>x1)*(f0+a*x1)

    def test_ESP(self,u_in,d_init,show_progress=True):

        u_in = self.check_data_in_comp(u_in)

        rand_dev = np.random.normal(0.,1.,(self.N))

        rand_dev *= d_init/np.linalg.norm(rand_dev)

        y0 = np.random.rand(self.N)*2.-1.
        y1 = y0[:] + rand_dev

        n_t = u_in.shape[0]

        dist = np.ndarray((n_t))

        dist[0] = np.linalg.norm(y0-y1)

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y0 = y0*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y0 + self.w_in @ u_in[t,:] - self.bias))
            y1 = y1*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y1 + self.w_in @ u_in[t,:] - self.bias))

            dist[t] = np.linalg.norm(y0-y1)

        logd = np.log(dist)

        logd = logd[np.where(logd!=-np.inf)]

        n_t = logd.shape[0]

        p,pc = curve_fit(self.esp_fit_func,np.array(range(n_t)),logd,[np.log(d_init),1.,100.])

        return dist,p,p[1]<0.


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

    def run_sample(self,u_in,show_progress=True,subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        y_rec = np.ndarray((n_rec,self.N))
        y_rec[0,:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] + self.w_in @ np.random.rand(self.data_dim_in) - self.bias))
        y = y_rec[0,:]

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + self.w_in @ u_in[t,:] - self.bias))

            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)

                y_rec[t_rec,:] = y

        return y_rec

    def run_hom_adapt_weight_mean_field(self,u_in,
                                            return_reservoir_rec=False,
                                            return_gain_rec=False,
                                            return_bias_rec=False,
                                            return_squ_act_target_rec=False,
                                            return_mu_y_rec=False,
                                            return_y_squ_rec=False,
                                            return_v_rec=False,
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

        if return_squ_act_target_rec:
            squ_act_target_rec = np.ndarray((n_rec,self.N))
            squ_act_target_rec[0,:] = self.sigm_act_target**2.

        if return_y_squ_rec:
            y_squ_rec = np.ndarray((n_rec,self.N))
            y_squ_rec[0,:] = np.ones((self.N))*.25
        y_squ = np.ones((self.N))*.25**2.

        if return_mu_y_rec:
            mu_y_rec = np.ndarray((n_rec,self.N))
            mu_y_rec[0,:] = np.zeros((self.N))
        mu_y = np.zeros((self.N))

        if return_v_rec:
            v_rec = np.ndarray((n_rec))
            v_rec[0] = 1.
        v = 1.

        W_T_squ = np.array(self.W.todense()).T**2.


        for t in tqdm(range(1,n_t)):
            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W.dot(y) + self.w_in @ u_in[t,:] - self.bias))


            y_squ += self.eps_trail_av_sigm_act*(y**2. - y_squ)

            mu_y += self.eps_trail_av_act*(y - mu_y)

            v = (W_T_squ*self.gain**2.).mean()*self.N

            self.sigm_act_target += self.eps_sigm_act_target*self.sigm_act_target*(1.-self.sigm_act_target)*(1.-v)

            self.gain += self.eps_gain * (self.sigm_act_target - sigm_y)
            self.gain = np.maximum(0.,self.gain)

            self.bias += self.eps_bias * (y - self.mu_act_target)


            if t%subsample_rec == 0:
                t_rec = int(t/subsample_rec)
                if return_reservoir_rec:
                    y_rec[t_rec,:] = y
                if return_bias_rec:
                    bias_rec[t_rec,:] = self.bias
                if return_gain_rec:
                    gain_rec[t_rec,:] = self.gain
                if return_sigm_act_target_rec:
                    sigm_act_target_rec[t_rec] = self.sigm_act_target
                if return_mu_y_rec:
                    mu_y_rec[t_rec,:] = mu_y
                if return_sigm_y_rec:
                    sigm_y_rec[t_rec,:] = sigm_y
                if return_v_rec:
                    v_rec[t_rec] = v


        returndat = []

        if return_reservoir_rec:
            returndat.append(y_rec)
        if return_bias_rec:
            returndat.append(bias_rec)
        if return_gain_rec:
            returndat.append(gain_rec)
        if return_sigm_act_target_rec:
            returndat.append(sigm_act_target_rec)
        if return_mu_y_rec:
            returndat.append(mu_y_rec)
        if return_sigm_y_rec:
            returndat.append(sigm_y_rec)
        if return_v_rec:
            returndat.append(v_rec)

        if len(returndat)==1:
            return returndat[0]
        elif len(returndat)>1:
            return returndat



    def run_hom_adapt_fisher(self,
                            u_in,
                            return_reservoir_rec=False,
                            return_mu_reservoir=False,
                            return_sigm_reservoir=False,
                            return_gain_rec=False,
                            return_bias_rec=False,
                            return_mu_ext=False,
                            return_sigm_ext=False,
                            return_mu_recurr=False,
                            return_sigm_recurr=False,
                            show_progress=True,
                            subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        if return_reservoir_rec:
            y_rec = np.ndarray((n_rec,self.N))
            y_rec[0,:] = np.tanh(self.gain*(self.W @ (np.random.rand(self.N)*2.-.5) + self.w_in @ u_in[0,:] - self.bias))
            y = y_rec[0,:]
            y_trail_av = y_rec[0,:]
        else:
            y = np.tanh(self.gain*(self.W @ (np.random.rand(self.N)*2.-.5) + self.w_in @ u_in[0,:] - self.bias))
            y_trail_av = np.array(y)

        if return_mu_reservoir:
            mu_y_rec = np.ndarray((n_rec,self.N))
            mu_y_rec[0,:] = y_rec[0,:]

        sigm_y_squ=np.ones((self.N))*10.**-3.
        if return_sigm_reservoir:
            sigm_y_rec = np.ndarray((n_rec,self.N))
            sigm_y_rec[0,:] = np.ones((self.N))*10.**-6.

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

            recurr_in = self.W @ y

            mu_ext = self.eps_trail_av_mu_ext*ext_in + (1.-self.eps_trail_av_mu_ext)*mu_ext

            sigm_ext_squ = self.eps_trail_av_sigm_ext*(ext_in - mu_ext)**2. + (1.-self.eps_trail_av_sigm_ext)*sigm_ext_squ

            sigm_ext = sigm_ext_squ**.5

            mu_recurr = self.eps_trail_av_mu_recurr*recurr_in + (1.-self.eps_trail_av_mu_ext)*mu_recurr

            sigm_recurr_squ = self.eps_trail_av_sigm_recurr*(recurr_in - mu_recurr)**2. + (1.-self.eps_trail_av_sigm_recurr)*sigm_recurr_squ

            sigm_recurr = sigm_recurr_squ**.5

            #self.sigm_act_target = ((1.5)**.5 * self.sigm_w/sigm_ext + 1.)**(-.5)



            #self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)

            X = recurr_in + ext_in

            if self.noise_level == 0:
                y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(X - self.bias))
            else:
                y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(X - self.bias + self.noise_level*np.random.normal(0.,1.,(self.N))))

            self.gain += self.eps_gain*(1-2.*y*X*self.gain)*(1.+2.*X**2.*(1.-y**2.)*self.gain**2.)

            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)
            sigm_y_squ += self.eps_trail_av_sigm_act*((y - y_trail_av)**2 - sigm_y_squ)

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
                if return_mu_recurr:
                    mu_recurr_rec[t_rec,:] = mu_recurr
                if return_sigm_recurr:
                    sigm_recurr_rec[t_rec,:] = sigm_recurr
                if return_mu_reservoir:
                    mu_y_rec[t_rec,:] = y_trail_av
                if return_sigm_reservoir:
                    sigm_y_rec[t_rec,:] = sigm_y_squ**.5

        returndat = []



        if return_reservoir_rec:
            returndat.append(y_rec)
        if return_mu_reservoir:
            returndat.append(mu_y_rec)
        if return_sigm_reservoir:
            returndat.append(sigm_y_rec)
        if return_bias_rec:
            returndat.append(bias_rec)
        if return_gain_rec:
            returndat.append(gain_rec)
        if return_mu_recurr:
            returndat.append(mu_recurr_rec)
        if return_sigm_recurr:
            returndat.append(sigm_recurr_rec)
        if return_mu_ext:
            returndat.append(mu_ext_rec)
        if return_sigm_ext:
            returndat.append(sigm_ext_rec)



        if len(returndat)==1:
            return returndat[0]

        elif len(returndat)>1:
            return returndat


    def run_hom_adapt_auto_target(self,
                                u_in,
                                return_reservoir_rec=False,
                                return_mu_reservoir=False,
                                return_sigm_reservoir=False,
                                return_gain_rec=False,
                                return_bias_rec=False,
                                return_mu_ext=False,
                                return_sigm_ext=False,
                                return_mu_recurr=False,
                                return_sigm_recurr=False,
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

        if return_mu_reservoir:
            mu_y_rec = np.ndarray((n_rec,self.N))
            mu_y_rec[0,:] = y_rec[0,:]

        sigm_y_squ=np.ones((self.N))*10.**-3.
        if return_sigm_reservoir:
            sigm_y_rec = np.ndarray((n_rec,self.N))
            sigm_y_rec[0,:] = np.ones((self.N))*10.**-6.

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

            recurr_in = self.W @ y

            mu_ext = self.eps_trail_av_mu_ext*ext_in + (1.-self.eps_trail_av_mu_ext)*mu_ext

            sigm_ext_squ = self.eps_trail_av_sigm_ext*(ext_in - mu_ext)**2. + (1.-self.eps_trail_av_sigm_ext)*sigm_ext_squ

            sigm_ext = sigm_ext_squ**.5

            mu_recurr = self.eps_trail_av_mu_recurr*recurr_in + (1.-self.eps_trail_av_mu_ext)*mu_recurr

            sigm_recurr_squ = self.eps_trail_av_sigm_recurr*(recurr_in - mu_recurr)**2. + (1.-self.eps_trail_av_sigm_recurr)*sigm_recurr_squ

            sigm_recurr = sigm_recurr_squ**.5

            self.sigm_act_target = ((1.5)**.5 * self.sigm_w/sigm_ext + 1.)**(-.5)

            self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)
            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            if self.noise_level == 0:
                y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + ext_in - self.bias))
            else:
                y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(self.W @ y + ext_in - self.bias + self.noise_level*np.random.normal(0.,1.,(self.N))))

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)
            sigm_y_squ += self.eps_trail_av_sigm_act*((y - y_trail_av)**2 - sigm_y_squ)

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
                if return_mu_recurr:
                    mu_recurr_rec[t_rec,:] = mu_recurr
                if return_sigm_recurr:
                    sigm_recurr_rec[t_rec,:] = sigm_recurr
                if return_mu_reservoir:
                    mu_y_rec[t_rec,:] = y_trail_av
                if return_sigm_reservoir:
                    sigm_y_rec[t_rec,:] = sigm_y_squ**.5

        returndat = {}

        if return_reservoir_rec:
            returndat['reservoir'] = y_rec
        if return_mu_reservoir:
            returndat['mu_reservoir'] = mu_y_rec
        if return_sigm_reservoir:
            returndat['sigm_reservoir'] = sigm_y_rec
        if return_bias_rec:
            returndat['bias'] = bias_rec
        if return_gain_rec:
            returndat['gain'] = gain_rec
        if return_mu_recurr:
            returndat['mu_recurr_in'] = mu_recurr_rec
        if return_sigm_recurr:
            returndat['sigm_recurr_in'] = sigm_recurr_rec
        if return_mu_ext:
            returndat['mu_ext_in'] = mu_ext_rec
        if return_sigm_ext:
            returndat['sigm_ext_in'] = sigm_ext_rec

        if len(returndat)==1:
            return returndat[0]

        elif len(returndat)>1:
            return returndat

    def run_adapt_RFLO(self,
                                u_in,
                                u_out,
                                alpha=1.,
                                alpha_w_out=1.,
                                T_w_o_learn = 2000,
                                T_skip_w_o_learn = 2000,
                                return_reservoir_rec=False,
                                return_gain_rec=False,
                                return_w_out_rec=False,
                                return_err_rec=False,
                                show_progress=True,
                                subsample_rec=1):

        u_in = self.check_data_in_comp(u_in)

        u_out = self.check_data_out_comp(u_out)

        n_t = u_in.shape[0]

        n_rec = int(n_t/subsample_rec)

        y = np.ndarray((self.N+1))
        y[0] = 1.

        if return_reservoir_rec:
            y_rec = np.ndarray((n_rec,self.N))
            y_rec[0,:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))
            y[1:] = y_rec[0,:]
        else:
            y[1:] = np.tanh(self.gain*(self.w_in @ u_in[0,:] - self.bias))

        if return_gain_rec:
            a_rec = np.ndarray((n_rec,self.N))

        if return_w_out_rec:
            w_out_rec = np.ndarray((n_rec,self.N+1))

        if return_err_rec:
            E_rec = np.ndarray((n_rec))

        y_rec_w_out_learn = np.zeros((T_w_o_learn,self.N+1))

        u_out_w_out_learn = np.zeros((T_w_o_learn,1))

        for t in tqdm(range(n_t)):

            X_r = self.W.dot(y[1:])

            X_e = self.w_in @ u_in[t,:]

            X = X_r + X_e

            y[1:] = np.tanh(self.gain*X)

            y_rec_w_out_learn[t%T_w_o_learn,:] = y[:]

            u_out_w_out_learn[t%T_w_o_learn,0] = u_out[t]

            O = (self.w_out @ y)[0]

            ### update readout weights

            #self.w_out[0,:] += -self.eps_LMS_out * (O-u_out[t]) * y

            #'''
            if t%T_skip_w_o_learn == 0 and t>=T_w_o_learn:

                self.w_out[0,:] = self.w_out[0,:]*(1.-alpha_w_out) + alpha_w_out * (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + self.reg_fact*np.eye(self.N+1)) @ y_rec_w_out_learn.T @ u_out_w_out_learn)[:,0]
            #'''

            ### update gains
            if t>=T_w_o_learn:

                delta_a = - self.eps_gain * (O-u_out[t]) * self.w_out[0,1:] * (1.-y[1:]**2.)*X_r

                self.gain += (1.-alpha)*delta_a + alpha*delta_a.mean()

                self.gain = np.maximum(self.gain,0.001)



            ####
            if t%subsample_rec == 0:

                t_rec = int(t/subsample_rec)

                if return_reservoir_rec:
                    y_rec[t_rec,:] = y[1:]
                if return_gain_rec:
                    a_rec[t_rec,:] = self.gain
                if return_w_out_rec:
                    w_out_rec[t_rec,:] = self.w_out
                if return_err_rec:
                    E_rec[t_rec] = (O - u_out[t])**2./2.


        w_out = (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + self.reg_fact*np.eye(self.N+1)) @ y_rec_w_out_learn.T @ u_out[t-T_w_o_learn+1:t+1,:])[:,0]


        returndat = {}

        if return_reservoir_rec:
            returndat['reservoir'] = y_rec
        if return_gain_rec:
            returndat['gain'] = a_rec
        if return_w_out_rec:
            returndat['w_out'] = w_out_rec
        if return_err_rec:
            returndat['err'] = E_rec

        if len(returndat) > 0:
            returndat['t_ax'] = t_ax = np.array(range(n_rec)) * subsample_rec

        if len(returndat)==1:
            return returndat[0]

        elif len(returndat)>1:
            return returndat




    '''
    def run_hom_adapt_auto(self,
                            u_in,
                            return_reservoir_rec=False,
                            return_mu_reservoir=False,
                            return_sigm_reservoir=False,
                            return_gain_rec=False,
                            return_bias_rec=False,
                            return_mu_ext=False,
                            return_sigm_ext=False,
                            return_mu_recurr=False,
                            return_sigm_recurr=False,
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

        if return_mu_reservoir:
            mu_y_rec = np.ndarray((n_rec,self.N))
            mu_y_rec[0,:] = y_rec[0,:]

        sigm_y_squ=np.ones((self.N))*10.**-3.
        if return_sigm_reservoir:
            sigm_y_rec = np.ndarray((n_rec,self.N))
            sigm_y_rec[0,:] = np.ones((self.N))*10.**-6.

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

            recurr_in = self.W @ y

            mu_ext = self.eps_trail_av_mu_ext*ext_in + (1.-self.eps_trail_av_mu_ext)*mu_ext

            sigm_ext_squ = self.eps_trail_av_sigm_ext*(ext_in - mu_ext)**2. + (1.-self.eps_trail_av_sigm_ext)*sigm_ext_squ

            sigm_ext = sigm_ext_squ**.5

            mu_recurr = self.eps_trail_av_mu_recurr*recurr_in + (1.-self.eps_trail_av_mu_ext)*mu_recurr

            sigm_recurr_squ = self.eps_trail_av_sigm_recurr*(recurr_in - mu_recurr)**2. + (1.-self.eps_trail_av_sigm_recurr)*sigm_recurr_squ

            sigm_recurr = sigm_recurr_squ**.5

            #self.sigm_act_target = ((1.5)**.5 * self.sigm_w/sigm_ext + 1.)**(-.5)

            self.sigm_act_target = (1.-(1+2.*(sigm_recurr_squ + sigm_ext_squ/self.sigm_w**2.))**(-.5))**.5

            self.gain += self.eps_gain * (self.sigm_act_target**2 - (y - y_trail_av)**2)
            self.gain = np.maximum(0.,self.gain)
            self.bias += self.eps_bias * (y - self.mu_act_target)

            y = y*(1.-self.alpha) + self.alpha*np.tanh(self.gain*(recurr_in + ext_in - self.bias))

            y_trail_av += self.eps_trail_av_act*(y - y_trail_av)
            sigm_y_squ += self.eps_trail_av_sigm_act*((y - y_trail_av)**2 - sigm_y_squ)

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
                if return_mu_recurr:
                    mu_recurr_rec[t_rec,:] = mu_recurr
                if return_sigm_recurr:
                    sigm_recurr_rec[t_rec,:] = sigm_recurr
                if return_mu_reservoir:
                    mu_y_rec[t_rec,:] = y_trail_av
                if return_sigm_reservoir:
                    sigm_y_rec[t_rec,:] = sigm_y_squ**.5

        returndat = {}

        if return_reservoir_rec:
            returndat['reservoir'] = y_rec
        if return_mu_reservoir:
            returndat['mu_reservoir'] = mu_y_rec
        if return_sigm_reservoir:
            returndat['sigm_reservoir'] = sigm_y_rec
        if return_bias_rec:
            returndat['bias'] = bias_rec
        if return_gain_rec:
            returndat['gain'] = gain_rec
        if return_mu_recurr:
            returndat['mu_recurr_in'] = mu_recurr_rec
        if return_sigm_recurr:
            returndat['sigm_recurr_in'] = sigm_recurr_rec
        if return_mu_ext:
            returndat['mu_ext_in'] = mu_ext_rec
        if return_sigm_ext:
            returndat['sigm_ext_in'] = sigm_ext_rec

        if len(returndat)==1:
            return returndat[0]

        elif len(returndat)>1:
            return returndat
    '''
    def get_params(self):

        dict = {"":self.N,
        "cf":self.cf,
        "sigm_w":self.sigm_w,
        "sigm_w_in":self.sigm_w_in,
        "alpha":self.alpha,
        "cf_w_in":self.cf_w_in,
        "data_dim_in":self.data_dim_in,
        "data_dim_out":self.data_dim_out,
        "reg_fact":self.reg_fact,
        "bias":self.bias,
        "gain":self.gain,
        "eps_gain":self.eps_gain,
        "eps_bias":self.eps_bias,
        "mu_act_target":self.mu_act_target,
        "sigm_act_target":self.sigm_act_target,
        "eps_trail_av_act":self.eps_trail_av_act,
        "eps_trail_av_sigm_act":self.eps_trail_av_sigm_act,
        "eps_trail_av_mu_ext":self.eps_trail_av_mu_ext,
        "eps_trail_av_sigm_ext":self.eps_trail_av_sigm_ext,
        "eps_trail_av_mu_recurr":self.eps_trail_av_mu_recurr,
        "eps_trail_av_sigm_recurr":self.eps_trail_av_sigm_recurr,
        "noise_level":self.noise_level}

        return dict
