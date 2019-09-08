#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

class RNN:

    def __init__(self,
                N=1000,
                cf=.1,
                cf_w_in=1.,
                sigm_w=1.,
                sigm_w_in=1.,
                sigm_w_out_init=.1,
                tau=1.,
                data_dim_in=1,
                data_dim_out=1,
                eps_a=0.0001,
                eps_w_out = 0.0001):

        self.N = N

        self.W = np.random.normal(0.,sigm_w/(N*cf)**.5,(N,N))*(np.random.rand(N,N) <= cf)
        self.W[range(N),range(N)] = 0.

        self.cf = (1.*(self.W!=0.)).sum()/N**2.

        self.sigm_w = sigm_w

        self.Wt = self.W.T

        if self.cf < .5:
            self.W = csr_matrix(self.W)
            self.sparse_W = True
        else:
            self.sparse_W = False

        self.data_dim_in = data_dim_in
        self.data_dim_out = data_dim_out

        self.w_in = np.random.normal(0.,sigm_w_in,(self.N,self.data_dim_in))*(np.random.rand(self.N,self.data_dim_in) <= cf_w_in)

        self.cf_w_in = (1.*(self.w_in!=0.)).sum()/(N*data_dim_in)

        self.w_out = np.random.rand(data_dim_out,self.N+1)-.5
        self.w_out[:,0] = 0.

        self.tau = tau

        self.a = np.ones(self.N)

        self.eps_a = eps_a
        self.eps_w_out = eps_w_out

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
                #sys.exit()
            return np.array([data]).T

        elif (len(data.shape)>2) or (data.shape[1] != self.data_dim_out):
            print("output dimensions do not fit!")
            #sys.exit()

        return data


    def learn_gain(self,
                    u_in,
                    u_out,
                   mode,
                   mode_w_out = "batch",
                   randomize_RTRL_matrix=False,
                   fix_gain_radius=False,
                  T_batch_w_out=None,
                    tau_batch_w_out = 1.,
                    tau_grad_desc_wout = 0.001,
                    reg_fact = 0.01,
                    T_stop_learn_wout = None,
                  T_skip_rec=10,
                              show_progress=True,
                  return_y = True,
                  return_X = True,
                  return_X_r = True,
                  return_X_e = True,
                  return_a = True,
                  return_dyda = True,
                  return_delta_a = True,
                  return_w_out = True,
                  return_Err = True,
                return_W = True,
                 ax_a=None,
                 ax_w_out=None,
                 ax_Err=None,
                 T_plot=500):


        modelist = ["local_grad_local_gain",
                        "local_grad_global_gain",
                        "global_grad_local_gain",
                        "global_grad_global_gain"]

        if not(mode in modelist):
            print("wrong mode argument!")
            return None
        else:
            mode = modelist.index(mode)

        modelist_w_out = ["batch","grad_descent"]

        if not(mode_w_out in modelist_w_out):
            print("wrong w_out mode argument!")
            return None
        else:
            mode_w_out = modelist_w_out.index(mode_w_out)


        if T_batch_w_out == None:
            T_batch_w_out = self.N*10

        u_in = self.check_data_in_comp(u_in)
        u_out = self.check_data_out_comp(u_out)

        if u_in.shape[0] == u_out.shape[0]:
            T = u_in.shape[0]
        else:
            print("length of time series do not match!")

        if T_stop_learn_wout == None:
            T_stop_learn_wout = T

        ### recording

        T_rec = int(T/T_skip_rec)


        if return_y:
            y_rec = np.ndarray((T_rec,self.N))
        if return_X_r:
            X_r_rec = np.ndarray((T_rec,self.N))
        if return_X_e:
            X_e_rec = np.ndarray((T_rec,self.N))
        if return_X:
            X_rec = np.ndarray((T_rec,self.N))
        if return_a:
            a_rec = np.ndarray((T_rec,self.N))
        if return_dyda:
            if mode == 0:
                dyda_rec = np.ndarray((T_rec,self.N))
            if mode == 1:
                dyda_rec = np.ndarray((T_rec,self.N))
            if mode == 2:
                dyda_rec = np.ndarray((T_rec,self.N,self.N))
            if mode == 3:
                dyda_rec = np.ndarray((T_rec,self.N))
        if return_delta_a:
            if mode == 0:
                delta_a_rec = np.ndarray((T_rec,self.N))
            if mode == 1:
                delta_a_rec = np.ndarray((T_rec))
            if mode == 2:
                delta_a_rec = np.ndarray((T_rec,self.N))
            if mode == 3:
                delta_a_rec = np.ndarray((T_rec))
        if return_w_out:
            w_out_rec = np.ndarray((T_rec,self.data_dim_out,self.N+1))
        if return_Err:
            Err_rec = np.ndarray((T_rec))


        if mode == 0:
            dyda = np.zeros((self.N))
            delta_a = np.zeros((self.N))

        if mode == 1:
            dyda = np.zeros((self.N))
            delta_a = 0.

        if mode == 2:
            dyda = np.zeros((self.N,self.N))
            delta_a = np.zeros((self.N))

        if mode == 3:
            dyda = np.zeros((self.N))
            delta_a = 0.




        y_rec_w_out_learn = np.zeros((T_batch_w_out,self.N+1))

        u_out_w_out_learn = np.zeros((T_batch_w_out,self.data_dim_out))
        ###

        y = np.random.rand(self.N+1)-.5
        y[0] = 0.


        if randomize_RTRL_matrix:
            '''
            W_RTRL = np.random.normal(0.,self.sigm_w/(self.N*self.cf)**.5,(self.N,self.N))*(np.random.rand(self.N,self.N) <= self.cf)
            W_RTRL[range(self.N),range(self.N)] = 0.
            '''
            #W_RTRL = np.eye(self.N)
            W_RTRL = np.zeros((self.N,self.N))

            W_RTRL_t = np.array(W_RTRL.T)

            if self.sparse_W:
                W_RTRL = csr_matrix(W_RTRL)
                #W_RTRL_t = csr_matrix(W_RTRL_t)
        else:
            if self.sparse_W:
                W_RTRL = csr_matrix(self.W)
                W_RTRL_t = np.array(self.W.todense()).T
            else:
                W_RTRL = np.array(self.W)
                W_RTRL_t = np.array(self.W.T)


        if fix_gain_radius:
            a_rad = np.linalg.norm(self.a)


        Vt = np.zeros((self.N))
        betaV = 0.995
        St = np.zeros((self.N))
        betaS = 0.99

        ### Init variables for better performance
        X_r = np.ndarray((self.N))
        X_e = np.ndarray((self.N))
        X = np.ndarray((self.N))

        err = np.ndarray((1,self.data_dim_out))
        O = np.ndarray((1,self.data_dim_out))

        phi_y = np.ndarray((self.N))
        #D_a_phi_y = np.eye((self.N))
        #D_a_phi_y = csr_matrix(D_a_phi_y)

        for t in tqdm(range(T),disable=not(show_progress)):
            '''
            if randomize_RTRL_matrix and t%10 == 0:
                W_RTRL = np.random.normal(0.,self.sigm_w/(self.N*self.cf)**.5,(self.N,self.N))*(np.random.rand(self.N,self.N) <= self.cf)
                #W_RTRL = np.ones((self.N,self.N))*self.sigm_w/(self.N)
                W_RTRL[range(self.N),range(self.N)] = 0.

                W_RTRL_t = np.array(W_RTRL.T)

                if self.sparse_W:
                    W_RTRL = csr_matrix(W_RTRL)
                    #W_RTRL_t = csr_matrix(W_RTRL_t)
            '''

            X_r[:] = self.W.dot(y[1:])

            X_e[:] = self.w_in.dot(u_in[t,:])

            X[:] = X_r + X_e

            #y[1:] = np.tanh(self.a*X_r + X_e)

            y[1:] = np.tanh(self.a*X)

            #'''
            if t <= T_stop_learn_wout:

                y_rec_w_out_learn[t%T_batch_w_out,:] = y[:]

                u_out_w_out_learn[t%T_batch_w_out,:] = u_out[t,:]
            #'''
            O[:] = self.w_out.dot(y)

            err[0,:] = O - u_out[t,:]

            if mode_w_out == 0:

                if t%T_batch_w_out == 0 and t>0. and t <= T_stop_learn_wout:
                    #print("updated wout")

                    #self.learn_w_out(u_in[t-T_batch_w_out:t],u_out[t-T_batch_w_out:t])
                    self.w_out[:] = self.w_out + (1./tau_batch_w_out)*( -self.w_out + (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + reg_fact*np.eye(self.N+1)) @ y_rec_w_out_learn.T @ u_out_w_out_learn).T)

            elif mode_w_out == 1:

                if t == T_batch_w_out:
                    self.w_out[:] = self.w_out + (1./tau_batch_w_out)*( -self.w_out + (np.linalg.inv(y_rec_w_out_learn.T @ y_rec_w_out_learn + reg_fact*np.eye(self.N+1)) @ y_rec_w_out_learn.T @ u_out_w_out_learn).T)
                elif t > T_batch_w_out:

                    self.w_out[:] -= self.eps_w_out * np.outer(err,y)


            phi_y[:] = (1.-y[1:]**2.)

            if mode == 0:
                dyda[:] = phi_y*X

            if mode == 1:
                dyda[:] = phi_y*X

            if mode == 2:
                #D_a_phi_y[range(self.N),range(self.N)] = self.a * phi_y

                dyda[:,:] = (W_RTRL_t.T * phi_y * self.a).T @ dyda

                dyda[range(self.N),range(self.N)] += phi_y*X

            if mode == 3:
                dyda[:] = phi_y*(X + self.a[0]*W_RTRL.dot(dyda))


            if t >= T_batch_w_out:



                if mode == 0:
                    delta_a[:] = -err.dot(self.w_out[:,1:])[0,:]*dyda
                if mode == 1:
                    delta_a = -err.dot(self.w_out[:,1:].dot(dyda))
                if mode == 2:
                    delta_a[:] = -err.dot(self.w_out[:,1:].dot(dyda))[0,:]
                if mode == 3:
                    delta_a = -err.dot(self.w_out[:,1:].dot(dyda))




                if self.eps_a > 0.:
                    #'''
                    ## Accelerated Grad. Desc.
                    #Vt = betaV*Vt + (1.-betaV)*delta_a
                    #St = betaS*St + (1.-betaS)*(delta_a**2.)

                    #self.a += self.eps_a*Vt
                    #self.a += self.eps_a*Vt/(St + 10.**-1.)**.5
                    #'''
                    self.a += self.eps_a*delta_a
                    #self.a += self.eps_a*(0.9*delta_a.mean() + 0.1*delta_a)
                    self.a = np.maximum(0.01,self.a)

                    if fix_gain_radius:
                        self.a *= a_rad / np.linalg.norm(self.a)

            ### recording

            if t%T_skip_rec == 0:



                t_rec = int(t/T_skip_rec)


                if return_y:

                    y_rec[t_rec,:] = y[1:]

                if return_X:
                    X_rec[t_rec,:] = X
                if return_X_r:
                    X_r_rec[t_rec,:] = X_r
                if return_X_e:
                    X_e_rec[t_rec,:] = X_e

                if mode == 0:
                    if return_dyda:
                        dyda_rec[t_rec,:] = dyda
                    if return_delta_a:
                        delta_a_rec[t_rec,:] = delta_a
                if mode == 1:
                    if return_dyda:
                        dyda_rec[t_rec,:] = dyda
                    if return_delta_a:
                        delta_a_rec[t_rec] = delta_a

                if mode == 2:
                    if return_dyda:
                        dyda_rec[t_rec,:,:] = dyda
                    if return_delta_a:
                        delta_a_rec[t_rec,:] = delta_a

                if mode == 3:
                    if return_dyda:
                        dyda_rec[t_rec,:] = dyda
                    if return_delta_a:
                        delta_a_rec[t_rec] = delta_a

                if return_a:
                    a_rec[t_rec,:] = self.a[:]

                    if ax_a != None and t%T_plot == 0:
                        t_ax = np.array(range(t_rec))*T_skip_rec
                        ax_a.clear()
                        ax_a.plot(t_ax,a_rec[:t_rec,:10])
                        ax_a.plot(t_ax,np.linalg.norm(a_rec[:t_rec,:],axis=1)/self.N**.5,'--',c='k')
                        plt.pause(0.01)


                if return_Err:
                    Err_rec[t_rec] = .5*(err**2.).sum()

                    if ax_Err != None and t%T_plot == 0:
                        t_ax = np.array(range(t_rec))*T_skip_rec
                        ax_Err.clear()
                        #t_rec_start = int(np.maximum(0,t_rec-100))
                        #ax_Err.plot(t_ax[t_rec_start:t_rec],Err_rec[t_rec_start:t_rec])
                        ax_Err.plot(t_ax,Err_rec[:t_rec])
                        ax_Err.set_yscale("log")
                        plt.pause(0.01)
                if return_w_out:
                    w_out_rec[t_rec,:,:] = self.w_out

                    if ax_w_out != None and t%T_plot == 0:
                        t_ax = np.array(range(t_rec))*T_skip_rec
                        ax_w_out.clear()
                        ax_w_out.plot(t_ax,w_out_rec[:t_rec,0,:10])
                        plt.pause(0.01)



        t_ax = np.array(range(T_rec))*T_skip_rec

        result = [t_ax]

        if return_y:
            result.append(y_rec)
        if return_X:
            result.append(X_r_rec+X_e_rec)
        if return_X_r:
            result.append(X_r_rec)
        if return_X_e:
            result.append(X_e_rec)
        if return_a:
            result.append(a_rec)
        if return_dyda:
            result.append(dyda_rec)
        if return_delta_a:
            result.append(delta_a_rec)
        if return_w_out:
            result.append(w_out_rec)
        if return_Err:
            result.append(Err_rec)
        if return_W:
            result.append(np.array(self.W.todense()))

        return result



    def learn_w_out(self,u_in,u_target,reg_fact=0.01,t_prerun=0):

        u_in = self.check_data_in_comp(u_in)
        u_target = self.check_data_out_comp(u_target)

        n_t = u_in.shape[0]

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.


        y[0,1:] = np.tanh(self.w_in @ u_in[0,:])


        for t in tqdm(range(1,n_t)):

            y[t,1:] = np.tanh(self.a*self.W.dot(y[t-1,1:]) + self.w_in @ u_in[t,:])

        self.w_out[:,:] = (np.linalg.inv(y[t_prerun:,:].T @ y[t_prerun:,:] + reg_fact*np.eye(self.N+1)) @ y[t_prerun:,:].T @ u_target[t_prerun:,:]).T

    def predict_data(self,data,return_reservoir_rec=False):

        data = self.check_data_in_comp(data)

        n_t = data.shape[0]

        u_in = data

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = np.tanh(self.w_in @ u_in[0,:])

        for t in tqdm(range(1,n_t)):

            y[t,1:] = np.tanh(self.a*self.W.dot(y[t-1,1:]) + self.w_in @ u_in[t,:])

        out = (self.w_out @ y.T).T
        if self.data_dim_out == 1:
            out = out[:,0]

        if return_reservoir_rec:
            return (out,y)
        else:
            return out
