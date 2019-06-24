

import numpy as np

from esn_module import esn

from multiprocessing.dummy import Pool
from multiprocessing import Lock

import time

path = "/mnt/ceph/fschubert/data/max_lyap_sweep/"

filename = "Entropy_gauss_input.npy"


# In[64]:


n_sweep_sigm_e = 30
n_sweep_sigm_t = 30

sigm_e = np.linspace(0.,1.5,n_sweep_sigm_e)
sigm_t = np.linspace(0.,.9,n_sweep_sigm_t)

sigm_t_ax = np.linspace(0.,sigm_t[-1]+(sigm_t[-1]-sigm_t[-2]),n_sweep_sigm_t+1)
sigm_e_ax = np.linspace(0.,sigm_e[-1]+(sigm_e[-1]-sigm_e[-2]),n_sweep_sigm_e+1)

n_t_adapt = 150000

n_t_sample = 100000

nbins = 50
bins = np.linspace(-1.,1.,nbins+1)

H_list = np.ndarray((n_sweep_sigm_e,n_sweep_sigm_t,1000))


# In[65]:


n_threads = 2

pool = Pool(n_threads)

parallel_it = []

lock_parall = Lock()

for k in range(n_sweep_sigm_e):
    for l in range(n_sweep_sigm_t):
        parallel_it.append([lock_parall,k,l])

t0 = time.time()
n_el = 0


def run_simulation(parameters):

    lock,k,l = parameters

    global n_el, f

    lock.acquire()
    n_el += 1

    t_el = (time.time()-t0)/60.

    t_rest = (n_sweep_sigm_t*n_sweep_sigm_e - n_el)*t_el/n_el

    if t_rest >= 60.:
        str_t_rest = str(int(t_rest/60.)) +  ":" + str(int(t_rest%60.)) + " h"
    else:
        str_t_rest = '{:.2f}'.format(t_rest) + " min"
    print(str(l+k*n_sweep_sigm_t)+"/"+str(n_sweep_sigm_t*n_sweep_sigm_e) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go")
    f.write(str(l+k*n_sweep_sigm_t)+"/"+str(n_sweep_sigm_t*n_sweep_sigm_e) + " " + '{:.2f}'.format((time.time()-t0)/60.) +  " minutes elapsed, approx. " + str_t_rest + " to go\n")
    f.flush()
    lock.release()


    ESN = esn(sigm_act_target=sigm_t[l],mu_act_target=np.random.normal(0.,0.01,(1000)))

    u_in = np.random.normal(0.,sigm_e[k],(n_t_adapt)) #(np.random.rand(n_t_adapt)<0.5)*2.*sigm_e[k]

    ESN.run_hom_adapt(u_in,show_progress=False)

    u_in = np.random.normal(0.,sigm_e[k],(n_t_sample)) #(np.random.rand(n_t_sample)<0.5)*2.*sigm_e[k]

    y_rec = ESN.run_sample(u_in,show_progress=False)

    y_hist = np.ndarray((ESN.N,nbins))

    H_list_temp = np.ndarray((ESN.N))

    for m in range(ESN.N):
        y_hist[m,:] = np.histogram(y_rec[:,m],bins=bins)[0]/n_t_sample
        ind = np.where(y_hist[m,:]!=0)[0]
        H_list_temp[m] = -(y_hist[m,:][ind]*np.log(y_hist[m,:][ind])).sum()

    return H_list_temp

f = open("sim_text_Gauss_Entropy.txt","a")

results = pool.map(run_simulation,parallel_it)

pool.close()
pool.join()

f.close()

for i,p in enumerate(parallel_it):

    k = p[1]
    l = p[2]

    H_list[k,l,:] = results[i]

'''
for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_sigm_t)):

        ESN = esn(sigm_act_target=sigm_t[l],mu_act_target=np.random.normal(0.,0.01,(1000)))

        u_in = (np.random.rand(n_t_adapt)<0.5)*2.*sigm_e[k]

        ESN.run_hom_adapt(u_in,show_progress=False)

        u_in = (np.random.rand(n_t_sample)<0.5)*2.*sigm_e[k]

        y_rec = ESN.run_sample(u_in,show_progress=False)

        y_hist = np.ndarray((ESN.N,nbins))

        for m in range(ESN.N):
            y_hist[m,:] = np.histogram(y_rec[:,m],bins=bins)[0]/n_t_sample
            ind = np.where(y_hist[m,:]!=0)[0]
            H_list[k,l,m] = -(y_hist[m,:][ind]*np.log(y_hist[m,:][ind])).sum()
'''


np.save(path+filename,H_list)
