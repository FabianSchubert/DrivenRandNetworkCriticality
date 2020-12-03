import numpy as np

from src.rnn import RNN

from src.testfuncs import gen_in_out_one_in_subs

from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

import sys

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input_type",
help='''specify four type of input (homogeneous_identical_binary,
homogeneous_independent_gaussian, heterogeneous_identical_binary,
heterogeneous_independent_gaussian)''',
choices=['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'])

parser.add_argument("adaptation_mode",
help='''specify the mode of adaptation: local or global''',
choices=['local','global'])


parser.add_argument("--record_adaptation_vars",
help='''whether to record and save the time course of gains and biases during adaptation''',
dest="record_adaptation",
action="store_true")
parser.add_argument("--no_record_adaptation_vars",
dest="record_adaptation",
action="store_false")
parser.set_defaults(record_adaptation_vars=False)

parser.add_argument("--record_r_a_est",
help='''Wheter to record and save an estimate of the spectral radius during adaptation''',
dest="record_r_a_est",
action="store_true")
parser.set_defaults(record_r_a_est=False)


parser.add_argument("--N",
help="number of neurons",
type=int,
default=500)

parser.add_argument("--n_samples",
help='''number of sample runs''',
type=int,
default=1)

parser.add_argument("--n_sweep_sigm_e",
help="number of sweep steps for external input variance",
type=int)

parser.add_argument("--n_sweep_r_t",
help="number of sweep steps for target spectral radius",
type=int,
default=30)

parser.add_argument("--cf_w",
help='''recurrent connection fraction''',
type=float,
default=.1)

parser.add_argument("--cf_w_in",
help='''external input connection fraction''',
type=float,
default=1.)

parser.add_argument("--eps_a",
help='''gain learning rate''',
type=float,
default=1e-3)

parser.add_argument("--eps_b",
help='''bias learning rate''',
type=float,
default=1e-3)

parser.add_argument("--eps_X_r_squ",
help='''adaptation rate for trailing average of squared recurrent membrane potential''',
type=float,
default=1e-2)

parser.add_argument("--T_run_adapt",
help="number of time steps for adaptation",
type=int,
default=25000)

parser.add_argument("--T_skip_rec",
help="number of time steps to skip for recording (meaningless if you do not record)",
type=int,
default=10)

parser.add_argument("--T_prerun_sample",
help="number of prerun time steps before recording a sample",
type=int,
default=100)

parser.add_argument("--T_run_sample",
help="time steps for recording a sample",
type=int,
default=1000)

parser.add_argument("--mu_y_target",
help='''activity target''',
type=float,
default=0.05)

parser.add_argument("--norm_flow",
help='''The adaptation rate for the gains is normalized by the x_r^2''',
dest="norm_flow",
action="store_true")
parser.add_argument("--no_norm_flow",
help="The adaptation rate for the gains is NOT normalized by the x_r^2",
dest="norm_flow",
action="store_false")
parser.set_defaults(norm_flow=True)


parser.add_argument("--a_init_offset",
help='''Initial offset for gain factors from the target spectral radius''',
type=float,
default=0.)

parser.add_argument("--save_dir",
help='''Directory for saving data. Base folder is the folder corresponding 
to the input type.''',
default='alt_hom_regulation')

parser.add_argument("--filename",
help='''Name of the simulation data file''')

args = parser.parse_args()

input_type = ['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'].index(args.input_type)

adaptation_mode = ['local','global'].index(args.adaptation_mode)

N = args.N
n_samples = args.n_samples

if args.n_sweep_sigm_e is None:
    n_sweep_sigm_e = 3
    sigm_e = np.array([0.,.5,1.5])
elif args.n_sweep_sigm_e == 1:
    n_sweep_sigm_e = 1
    sigm_e = np.array([.5])
else:
    n_sweep_sigm_e = args.n_sweep_sigm_e
    sigm_e = np.linspace(0.,1.,n_sweep_sigm_e)

n_sweep_r_t = args.n_sweep_r_t
cf_w = args.cf_w
cf_w_in = args.cf_w_in
eps_a = args.eps_a
eps_b = args.eps_b
eps_X_r_squ = args.eps_X_r_squ
T_run_adapt = args.T_run_adapt
T_prerun_sample = args.T_prerun_sample
T_run_sample = args.T_run_sample
T_skip_rec = args.T_skip_rec
T_rec = int(T_run_adapt/T_skip_rec)
mu_y_target = args.mu_y_target

norm_flow = args.norm_flow

a_init_offset = args.a_init_offset

record_adaptation = args.record_adaptation_vars
record_r_a_est = args.record_r_a_est

if args.filename == None:
    filename = ('param_sweep_alt_hom_regulation_'
    +args.adaptation_mode
    +'_'+str(datetime.now().isoformat())+'.npz')
else:
    filename = args.filename

if n_sweep_sigm_e == 1:
    sigm_e = np.array([0.05])

r_t = np.linspace(0.,2.,n_sweep_r_t)

if n_sweep_r_t == 1:
    r_t = np.array([.1])

W_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,N,N))
w_in_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,N))

y_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,T_run_sample,N))
X_r_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,T_run_sample,N))
X_e_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,T_run_sample,N))

if record_adaptation:
    a_store = np.ndarray((n_samples,T_rec,n_sweep_sigm_e,n_sweep_r_t,N))
    b_store = np.ndarray((n_samples,T_rec,n_sweep_sigm_e,n_sweep_r_t,N))

else:
    a_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,N))
    b_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t,N))

if record_r_a_est:
    r_a_est_store = np.ndarray((n_samples,T_rec,n_sweep_sigm_e,n_sweep_r_t))
else:
    r_a_est_store = np.ndarray((n_samples,n_sweep_sigm_e,n_sweep_r_t))

for n in tqdm(range(n_samples)):
    for k in tqdm(range(n_sweep_sigm_e),leave=False):
        for l in tqdm(range(n_sweep_r_t),leave=False):

            W = np.random.normal(0.,1./(cf_w*N)**.5,(N,N)) * (np.random.rand(N,N) <= cf_w)
            W[range(N),range(N)] = 0.

            W = W/np.max(np.abs(np.linalg.eigvals(W)))

            W_store[n,k,l,:,:] = W

            a = np.ones((N))*(r_t[l] + a_init_offset)

            b = np.zeros((N))

            if sigm_e[k] > 0.:
                if (input_type in [0,1]):
                    w_in = np.ones(N) * (np.random.rand(N) <= cf_w_in) /cf_w_in**.5
                else:
                    w_in = np.random.normal(0.,1.,(N)) * (np.random.rand(N) <= cf_w_in) /cf_w_in**.5
            else:
                w_in = np.zeros((N))

            w_in_store[n,k,l,:] = w_in

            y = np.ndarray((N))
            X_r = np.ndarray((N))
            X_e = np.ndarray((N))
            #X_e = (w_in @ u_in).T
            #X_e = np.random.normal(0.,1.,(T,N)) * w_in[:,0]
            #X_e = np.random.normal(0.,.25,(T,N))


            ### first time step
            if input_type in [0,2]:
                X_e[:] = w_in * (2.*(np.random.rand() <= 0.5) - 1.) * sigm_e[k]
            else:
                X_e[:] = w_in * np.random.normal(0.,1.,(N)) * sigm_e[k]
            X_r[:] = (np.random.rand(N)-.5)
            y[:] = np.tanh(X_r[:] + X_e[:])

            ##trailing average of X_r**2 for adjusting the learning rate
            X_r_squ_av = (X_r**2.).mean()

            if record_adaptation:
                a_store[n,0,k,l,:] = a
                b_store[n,0,k,l,:] = b
            
            W_var = W.var(axis=1)
                        
            ### adaptation part
            for t in tqdm(range(1,T_run_adapt),leave=False):

                y_prev = y[:]

                X_r[:] = a[:] * (W @ y[:])

                if input_type in [0,2]:
                    X_e[:] = w_in * (2.*(np.random.rand() <= 0.5) - 1.) * sigm_e[k]
                else:
                    X_e[:] = w_in * np.random.normal(0.,1.,(N)) * sigm_e[k]

                y[:] = np.tanh(X_r + X_e - b)

                #y_squ_targ = 1.-1./(1.+2.*Var_y.mean() + 2.*Var_X_e)**.5

                #a = a + eps_a * a * ((y**2.).mean() - (X_r**2.).mean())

                X_r_squ_av += eps_X_r_squ*((X_r**2.).mean() - X_r_squ_av)

                if adaptation_mode == 0:
                    da = eps_a * a * (r_t[l]**2.*y_prev**2. - X_r**2.)
                else:
                    da = eps_a * a * (r_t[l]**2.*(y_prev**2.).mean() - (X_r**2.).mean())

                if norm_flow:
                    da /= X_r_squ_av.mean()

                a = a + da

                #if adaptation_mode == 0:
                    #a = a + eps_a * a * (r_t[l]**2. * y_prev**2. - X_r**2.)
                #    a = a + eps_a * a * (r_t[l]**2. * y_prev**2. - X_r**2.)/X_r_squ_av
                #else:
                #    #a = a + eps_a * a * (r_t[l]**2. * (y_prev**2.).mean() - (X_r**2.).mean())
                #    a = a + eps_a * a * (r_t[l]**2. * (y_prev**2.).mean() - (X_r**2.).mean())/X_r_squ_av
                #a = a + eps_a * (W_av @ (y_prev**2.) - X_r**2.)
                #a = a + eps_a * ((y**2.) - (X_r**2.))
                b = b + eps_b * (y - mu_y_target)

                a = np.maximum(0.001,a)

                if t%T_skip_rec == 0:
                    
                    t_rec = int(t/T_skip_rec)
                    
                    if record_adaptation:
                        a_store[n,t_rec,k,l,:] = a
                        b_store[n,t_rec,k,l,:] = b
                    
                    if record_r_a_est:
                        r_a_est_store[n,t_rec,k,l] = (a**2.*W_var).sum()**.5

            if not(record_adaptation):

                a_store[n,k,l,:] = a
                b_store[n,k,l,:] = b
            
            if not(record_r_a_est):
                r_a_est_store[n,k,l] = (a**2.*W_var).sum()**.5


            ###testing part
            rnn = RNN(N=N,y_mean_target=mu_y_target)
            rnn.W[:,:] = W
            rnn.a_r[:] = a
            rnn.b[:] = b
            rnn.w_in[:,0] = w_in
            '''
            for tau in range(tau_max):

                u_in_learn,u_out_learn = gen_in_out_one_in_subs(T_run_learn+T_prerun,tau)
                u_in_learn *= sigm_e[k]

                rnn.learn_w_out_trial(u_in_learn,u_out_learn,reg_fact=.01,show_progress=False,T_prerun=T_prerun)

                u_in_test,u_out_test = gen_in_out_one_in_subs(T_run_test+T_prerun,tau)
                u_in_test *= sigm_e[k]

                u_out_pred = rnn.predict_data(u_in_test)

                MC[n,k,l,tau] = np.corrcoef(u_out_test[T_prerun:],u_out_pred[T_prerun:])[0,1]**2.
            '''
            ######################################
            if input_type in [0,2]:

                u_in_sample,u_out_sample = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,0)
                u_in_sample *= sigm_e[k]

                y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=False)


            elif input_type == 1:

                #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
                y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e[k],T=T_run_sample+T_prerun_sample,show_progress=False)

            else:

                sigm_e_dist = np.abs(rnn.w_in[:,0]) * sigm_e[k]

                #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
                y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e_dist,T=T_run_sample+T_prerun_sample,show_progress=False)
            ######################################

            y_store[n,k,l,:,:] = y_res[T_prerun_sample:,:]
            X_r_store[n,k,l,:,:] = X_r_res[T_prerun_sample:,:]
            X_e_store[n,k,l,:,:] = X_e_res[T_prerun_sample:,:]

savefold = os.path.join(DATA_DIR,
                        args.input_type+'_input_ESN',
                        args.save_dir)

if not(os.path.isdir(savefold)):
    os.makedirs(savefold)

savepath = os.path.join(savefold, filename)

np.savez(savepath,
        r_t=r_t,
        sigm_e=sigm_e,
        W=W_store,
        a=a_store,
        b=b_store,
        r_a_est=r_a_est_store,
        w_in=w_in,
        y=y_store,
        X_r=X_r_store,
        X_e=X_e_store,
        T_skip=T_skip_rec)

'''
np.savez(os.path.join(DATA_DIR, args.input_type+'_input_ESN/performance_sweep/param_sweep_alt_hom_regulation_performance_test.npz'),
        r_a=r_a,
        sigm_e=sigm_e,
        W=W_store,
        a=a_store,
        b=b_store,
        MC=MC,
        w_in=w_in)
'''
