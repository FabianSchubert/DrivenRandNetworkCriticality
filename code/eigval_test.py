#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 1000

'''
def gen_rand(n,mu,sigm):

    return np.tanh(np.random.normal(mu,sigm,(n)))
'''

W = np.random.normal(0.,1./n**.5,(n,n))
W[range(n),range(n)] = 0.

D = np.zeros((n,n))
D[range(n),range(n)] = np.random.rand(n)*.5

l = np.linalg.eigvals(D @ W)
lW = np.linalg.eigvals(W)

plt.plot(lW.real,lW.imag,'.')
plt.plot(l.real,l.imag,'.',alpha=0.5)

plt.show()
