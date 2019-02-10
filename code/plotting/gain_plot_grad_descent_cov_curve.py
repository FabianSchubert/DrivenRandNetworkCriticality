#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('matplotlibrc')

x = np.linspace(-.9999,.9999,10000)
y = np.arctanh(x)*(1.-x**2)

x_turn = 0.64792

textwidth = 5.5532
std_figsize = (textwidth*0.7, textwidth * 0.4)

fig, ax = plt.subplots(1,1,figsize=std_figsize)

rect = patches.Rectangle((-x_turn,-.5),2*x_turn,1.,linewidth=0,facecolor='#CCCCCC')
ax.add_patch(rect)

ax.plot(x,y)

ax.set_xlim([-1.,1.])
ax.set_ylim([-.5,.5])

ax.grid()

ax.set_xlabel("y")
ax.set_ylabel("$\\mathrm{arctanh}\\left(y\\right)\\left(1-y^2\\right)$")

fig.tight_layout()

fig.savefig("../../plots/gain_grad_descent.png", dpi=300)
fig.savefig("../../plots/gain_grad_descent.pdf")

plt.show()
