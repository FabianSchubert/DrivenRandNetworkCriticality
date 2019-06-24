#!/usr/bin/env python3

import numpy as np
from scipy.stats import linregress
from tqdm import tqdm
import pdb

from esn_module import esn


def test_memory_cap(ESN,t_back_max,n_learn_samples,break_low_threshold,thresh_av_wind,input_gen):

    MC = np.ndarray((t_back_max))

    for t_back in tqdm(range(t_back_max)):

        u_in = input_gen(n_learn_samples)
        u_in_test = input_gen(n_learn_samples)

        u_out = np.ndarray((n_learn_samples))
        u_out_test= np.ndarray((n_learn_samples))

        if t_back==0:
            u_out[:] = u_in[:]
            u_out_test[:] = u_in_test[:]
        else:
            u_out[t_back:] = u_in[:-t_back]
            u_out[:t_back] = input_gen(t_back)

            u_out_test[t_back:] = u_in_test[:-t_back]
            u_out_test[:t_back] = input_gen(t_back)

        ESN.learn_w_out(u_in,u_out)



        u_out_test_predict = ESN.predict_data(u_in_test)

        MC[t_back-1] = np.corrcoef(u_out_test,u_out_test_predict)[0,1]**2.

        if t_back > thresh_av_wind:
            if MC[t_back-thresh_av_wind:t_back].mean() <= break_low_threshold:
                length_t = t_back+1
                break

    return MC[:length_t],MC[:length_t].sum()

'''
def test_memory_cap(W,gain,bias,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,input_gen,reg_fact):

    n = W.shape[0]

    MC = np.ndarray((t_back_max))

    length_t = t_back_max

    for t_back in tqdm(range(t_back_max)):

        w_in = np.random.normal(0.,1.,(n))



        ###### Fitting

        # input signal u
        u = np.ndarray((n_learn_samples+t_back))
        # echo state
        x = np.ndarray((n_learn_samples,n+1))# use n+1 to add a bias
        # target readout
        y = np.ndarray((n_learn_samples,1))

        t = 0


        u[t] = input_gen(t)
        x_prerun = np.tanh(u[t]*w_in)

        for t in range(1,t_back):
            u[t] = input_gen(t)
            x_prerun = np.tanh(gain*(np.dot(W,x_prerun)+u[t]*w_in - bias))

        for t in range(n_learn_samples):
            u[t+t_back] = input_gen(t+t_back)
            y[t] = u[t]
            if t == 0:
                x[t,1:] = np.tanh(gain*(np.dot(W,x_prerun) + u[t+t_back]*w_in - bias))
            else:
                x[t,1:] = np.tanh(gain*(np.dot(W,x[t-1,1:]) + u[t+t_back]*w_in - bias))

        x[:,0] = 1. #for bias
        #pdb.set_trace()
        ################# ridge regression
        w_out_est = np.linalg.inv(x.T @ x + reg_fact*np.eye(n+1)) @ x.T @ y
        #w_out_est = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x) + reg_fact*np.eye(n+1)),x.T),y)

        ###### Testing

        # input signal u
        u = np.ndarray((n_learn_samples+t_back))
        # echo state
        x = np.ndarray((n_learn_samples,n+1))# use n+1 to add a bias
        # target readout
        y = np.ndarray((n_learn_samples,1))

        t = 0

        u[t] = input_gen(t)
        x_prerun = np.tanh(u[t]*w_in)

        for t in range(1,t_back):
            u[t] = input_gen(t)
            x_prerun = np.tanh(gain*(np.dot(W,x_prerun)+u[t]*w_in - bias))

        for t in range(n_learn_samples):
            u[t+t_back] = input_gen(t+t_back)
            y[t] = u[t]
            if t == 0:
                x[t,1:] = np.tanh(gain*(np.dot(W,x_prerun) + u[t+t_back]*w_in - bias))
            else:
                x[t,1:] = np.tanh(gain*(np.dot(W,x[t-1,1:]) + u[t+t_back]*w_in - bias))

        x[:,0] = 1. #for bias


        y_est = x @ w_out_est

        #pdb.set_trace()
        MC[t_back] = np.cov(y_est[:,0],y[:,0])[1,0]**2/(y_est.var()*y.var())

        if t_back > tresh_av_wind:
            if MC[t_back-tresh_av_wind:t_back].mean() <= break_low_threshold:
                length_t = t_back+1
                break

    return MC[:length_t],MC[:length_t].sum()#, x, w_out_est, y
'''


def test_echo_state_prop(W,gain,bias,t_run,init_d,threshold,input_gen,**kwargs):

    n = W.shape[0]

    rand_dev = np.random.normal(0.,1.,(n))
    rand_dev *= init_d/np.linalg.norm(rand_dev)

    x = np.ndarray((2,n))

    x[0,:] = np.random.normal(0.,1.,(n))

    if kwargs is not None:
        for key, value in kwargs.items():
            if key == "x_init":
                x[0,:] = value


    x[1,:] = x[0,:] + rand_dev

    d_rec = np.ndarray((t_run))

    for t in range(t_run):

        d_rec[t] = np.linalg.norm(x[0,:]-x[1,:])

        u = input_gen(t)

        x[0,:] = np.tanh(gain*(np.dot(W,x[0,:]) + u - bias))
        x[1,:] = np.tanh(gain*(np.dot(W,x[1,:]) + u - bias))

    #for t in range(10,t_run):
    #   slope, intercept, r, p, stderr = linregress(np.array(range(t)),np.log(d_rec[:t]))
    #    if stderr >= linfit_err_th:
    #        break

    return 1.*(d_rec[-1]<threshold), d_rec


def gen_in_out_one_in_subs(T,tau):
    inp = np.ndarray((T))

    inp = (np.random.rand(T) < .5)*1.

    outp = np.ndarray((T))

    for k in range(T):

        if k - tau - 1 < 0:
            outp[k] = (np.random.rand() < .5)*1.
        else:
            outp[k] = (inp[k-tau] != inp[k-tau-1])*1.

    return inp, outp

'''
def test_XOR(W,gain,bias,t_back_max,n_learn_samples,break_low_threshold,tresh_av_wind,sigm_in,reg_fact):

    MC = np.zeros((t_back_max))

    length_t = t_back_max

    for t_back in range(t_back_max):

        inp_xor, out_xor = gen_in_out_one_in_subs(n_learn_samples,t_back)
        inp_xor_test, out_xor_test = gen_in_out_one_in_subs(n_learn_samples,t_back)

        inp_xor = (inp_xor -.5)*2.*sigm_in
        inp_xor_test = (inp_xor_test -.5)*2.*sigm_in

        N = W.shape[0]

        ESN = esn(N=N,
              data_dim_in=1,
              data_dim_out=1,
              cf_w_in=1.,
              cf=1.,
              reg_fact=reg_fact,
              spec_rad=1.,
              alpha=1.,
              bias = bias,
              gain = gain)

        ESN.W = W

        ESN.learn_w_out(inp_xor,out_xor,t_prerun=200,show_progress=False)

        out_xor_test_pr = ESN.predict_data(inp_xor_test,return_reservoir_rec=False,show_progress=False)

        MC[t_back] = np.corrcoef(out_xor_test,out_xor_test_pr)[0,1]**2.

        if t_back > tresh_av_wind:
            if MC[t_back-tresh_av_wind:t_back].mean() <= break_low_threshold:
                length_t = t_back+1
                break

    return MC[:length_t],MC[:length_t].sum()
'''

def test_XOR(ESN,t_back_max,n_learn_samples,break_low_threshold,thresh_av_wind,sigm_in,show_progress=True):

    MC = np.zeros((t_back_max))

    length_t = t_back_max

    for t_back in tqdm(range(t_back_max),disable=not(show_progress)):

        inp_xor, out_xor = gen_in_out_one_in_subs(n_learn_samples,t_back)
        inp_xor_test, out_xor_test = gen_in_out_one_in_subs(n_learn_samples,t_back)

        inp_xor = (inp_xor -.5)*2.*sigm_in
        inp_xor_test = (inp_xor_test -.5)*2.*sigm_in

        ESN.learn_w_out(inp_xor,out_xor,show_progress=False)

        out_xor_test_pr = ESN.predict_data(inp_xor_test,return_reservoir_rec=False,show_progress=False)

        MC[t_back] = np.corrcoef(out_xor_test,out_xor_test_pr)[0,1]**2.

        if t_back > thresh_av_wind:
            if MC[t_back-thresh_av_wind:t_back].mean() <= break_low_threshold:
                length_t = t_back+1
                break

    return MC[:length_t],MC[:length_t].sum()




if __name__ == "__main__":

    N = 200

    def gen_input(t):
        return (np.random.rand()-.5)*1.

    def gen_input_multdim(t):
        return (np.random.rand(N)-.5)*1.

    W = np.random.normal(0.,1./N**.5,(N,N))
    W[range(N),range(N)] = 0.

    n_g = 50
    g = np.linspace(0.5,2.,n_g)

    MC_sweep = np.ndarray((n_g))

    t_run_echo_state_test = 1000
    d_rec_sweep = np.ndarray((n_g,t_run_echo_state_test))
    ESP_sweep = np.ndarray((n_g))

    for k in tqdm(range(n_g)):
        MC,MC_agg = test_memory_cap(W*g[k],100,5000,gen_input,0.1)
        MC_sweep[k] = MC_agg

        ESP_sweep[k], d_rec_sweep[k,:] = test_echo_state_prop(W*g[k],t_run_echo_state_test,0.5,gen_input_multdim,.01)

    import matplotlib.pyplot as plt

    #plt.plot(x @ w, y, '.')

    #plt.plot([-1.,1.],[-1.,1.])
    plt.figure()
    plt.plot(g,MC_sweep)
    for k in range(1,n_g):
        if ESP_sweep[k] == 0 and ESP_sweep[k-1] == 1:
            ESP_trans = g[k]
            break
    plt.plot([ESP_trans,ESP_trans],[MC_sweep.min(),MC_sweep.max()],'--',c="k")


    plt.figure()
    for k in range(n_g):
        plt.plot(np.log(d_rec_sweep[k,:]),c=(1.*k/n_g,0.,1.-1.*k/n_g))
        #plt.plot(np.array(range(1000))*slope_d_sweep[k] + intercept_d_sweep[k],c=(1.*k/n_g,0.,1.-1.*k/n_g))
    plt.show()

    pdb.set_trace()
