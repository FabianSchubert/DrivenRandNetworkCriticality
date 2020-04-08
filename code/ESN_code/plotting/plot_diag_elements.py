#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *

import os

N = 500

W = np.random.normal(0.,1.,(N,N))
W[range(N),range(N)] = 0.

#optionally, randomize rows:
#W = (W.T * np.random.rand(N)).T

W /= np.abs(np.linalg.eigvals(W)).max()

l, w = np.linalg.eig(W)
idx_sort = np.argsort(np.abs(l))
l = l[idx_sort]
w = w.T[idx_sort].T

U = W.T @ W

sigm_squ,S_u = np.linalg.eig(U)
idx_sort = np.argsort(sigm_squ)
sigm_squ = sigm_squ[idx_sort]
S_u = S_u.T[idx_sort].T

C_yy = np.zeros((N,N))
C_yy[range(N),range(N)] = np.random.rand(N)

fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

ax.plot(C_yy.diagonal(),'.',
        markersize=7,
        label=r'$\widehat{C}_{\rm yy}$')
ax.plot((S_u.T @ C_yy @ S_u).diagonal(),'.',
        markersize=7,
        label=r'$\widehat{S}_u^\dagger'
            +r'\widehat{C}_{\rm yy}'
            +r'\widehat{S}_u$')
ax.plot([0,N-1],[C_yy.diagonal().mean()]*2,
        '--',
        label=r'$\left\langle'
        +r'\left(\widehat{C}_{\rm yy}\right)_{kk}'
        +r'\right\rangle$',
        c='k',lw=2)

ax.legend(frameon=True)
ax.set_xlabel("$k$")
ax.set_ylabel("$k$th Diagonal Entry")

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'diag_elements.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'diag_elements.png'),dpi=300)

plt.show()
