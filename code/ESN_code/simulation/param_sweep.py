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

parser.add_argument("--N",
help="number of neurons",
type=int,
default=500)

parser.add_argument("--n_sweep_sigm_e",
help="number of sweep steps for external input variance",
type=int)

parser.add_argument("--n_sweep_sigm_t",
help="number of sweep steps for target output variance",
type=int,
default=30)

parser.add_argument("--T_run_adapt",
help="number of time steps for adaptation",
type=int,
default=200000)

parser.add_argument("--T_prerun_sample",
help="number of prerun time steps before recording a sample",
type=int,
default=100)

parser.add_argument("--T_run_sample",
help="time steps for recording a sample",
type=int,
default=1000)

parser.add_argument("--y_mean_target",
help="target activity",
type=float,
default=0.05)

args = parser.parse_args()

input_type = ['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'].index(args.input_type)

N = args.N
if args.n_sweep_sigm_e is None:
    n_sweep_sigm_e = 3
    sigm_e = np.array([0.,.5,1.5])
elif args.n_sweep_sigm_e == 1:
    n_sweep_sigm_e = 1
    sigm_e = np.array([.5])
else:
    n_sweep_sigm_e = args.n_sweep_sigm_e
    sigm_e = .5*np.array(range(n_sweep_sigm_e))

n_sweep_sigm_t = args.n_sweep_sigm_t
if n_sweep_sigm_t == 1:
    sigm_t = np.array([0.5])
else:
    sigm_t = np.linspace(0.,0.9,n_sweep_sigm_t)

T_run_adapt = args.T_run_adapt
T_prerun_sample = args.T_prerun_sample
T_run_sample = args.T_run_sample
y_mean_target = args.y_mean_target

#####################################

y = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))
X_r = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))
X_e = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,T_run_sample,N))

W = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N,N))
a = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))
b = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,N))

####################################

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_sigm_t)):

        rnn = RNN(N=N,y_mean_target=y_mean_target,y_std_target=sigm_t[l])

        ##################

        if input_type == 0:

            rnn.w_in = np.ones((rnn.N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]

            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000)

            #run test sample
            u_in_sample,u_out_sample = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,0)
            u_in_sample *= sigm_e[k]

            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=True)


        elif input_type == 1:

            rnn.w_in = np.ones((rnn.N,1))

            adapt = rnn.run_hom_adapt(u_in=None,
                                    sigm_e=sigm_e[k],T=T_run_adapt,T_skip_rec=1000)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e[k],T=T_run_sample+T_prerun_sample,show_progress=True)

        elif input_type == 2:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]

            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000)

            #run test sample
            u_in_sample,u_out_sample = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,0)
            u_in_sample *= sigm_e[k]

            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=True)

        else:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            sigm_e_dist = np.abs(rnn.w_in[:,0]) * sigm_e[k]

            adapt = rnn.run_hom_adapt(u_in=None,sigm_e=sigm_e_dist,T=T_run_adapt,T_skip_rec=1000)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e_dist,T=T_run_sample+T_prerun_sample,show_progress=True)

        ####################################

        y[k,l,:,:] = y_res[T_prerun_sample:,:]
        X_r[k,l,:,:] = X_r_res[T_prerun_sample:,:]
        X_e[k,l,:,:] = X_e_res[T_prerun_sample:,:]

        W[k,l,:,:] = rnn.W
        a[k,l,:] = rnn.a_r
        b[k,l,:] = rnn.b


################################

if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type+'_input_ESN/gains_sweep'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type+'_input_ESN/gains_sweep'))

np.savez(os.path.join(DATA_DIR, args.input_type+'_input_ESN/gains_sweep/param_sweep_'+str(datetime.now().isoformat())+'.npz'),
        sigm_t=sigm_t,
        sigm_e=sigm_e,
        y=y,
        X_r=X_r,
        X_e=X_e,
        W=W,
        a=a,
        b=b)
