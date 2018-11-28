#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from tqdm import tqdm

### Parameters
N_net_def = 500
N_in_def = 10

cf_net_def = 0.1

std_conn_def = 1.

std_in_def = 1.

var_act_target_def = 0.33**2

mu_gain_def = 0.0005

n_t_def = 50000

t_ext_off_def = 25000


DN = driven_net(N_net_def,
            cf_net_def,
            std_conn_def,
            std_in_def,
            var_act_target_def,
            mu_gain_def,
            n_t_def,
            t_ext_off_def)

DN.run_sim()

### Plotting

t_ax = np.array(range(n_t_def-100,n_t_def))

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

#=====================================
fig_act_trans, ax_act_trans = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

t_pm = 100

t_ax_trans = np.array(range(t_ext_off_def - t_pm, t_ext_off_def + t_pm))

ax_act_trans.plot(t_ax_trans,DN.x_net_rec[t_ext_off_def - t_pm:t_ext_off_def + t_pm,:5])
ax_act_trans.set_xlabel("Time Step")
ax_act_trans.ticklabel_format(axis='x', style='sci', useOffset=t_ext_off_def, useMathText=True)
ax_act_trans.set_ylabel("Recurrent Activity")

ax_act_trans.set_title("A",{'fontweight' : 'bold'}, loc="left")

fig_act_trans.tight_layout()
fig_act_trans.savefig("../plots/act_trans.png", dpi=300)
######################################

#=====================================
fig_act, ax_act = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

ax_act.plot(t_ax,DN.x_net_rec[-100:,:5])
ax_act.set_xlabel("Time Step")
ax_act.ticklabel_format(axis='x', style='sci', useOffset=n_t_def-100, useMathText=True)
ax_act.set_ylabel("Recurrent Activity")

ax_act.set_title("E",{'fontweight' : 'bold'}, loc="left")

fig_act.tight_layout()
fig_act.savefig("../plots/act.png", dpi=300)
'''
ax_act.set_title("Sample of Population Activity for the last 100 Steps")
'''
#####################################

#====================================
fig_gain, ax_gain = plt.subplots(figsize=std_figsize,dpi=dpi_screen)

ax_gain.plot(DN.gain_rec[:,::10])
ax_gain.set_xlabel("Time Step")
ax_gain.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
ax_gain.set_ylabel("$g_i$")

ax_gain.set_title("B",{'fontweight' : 'bold'}, loc="left")

fig_gain.tight_layout()
fig_gain.savefig("../plots/gain.png", dpi=300)
#####################################

#====================================
fig_var, ax_var = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

ax_var.plot(DN.var_mean_rec)
ax_var.set_xlabel("Time Step")
ax_var.ticklabel_format(axis='x', style='sci', scilimits=(4,4), useMathText=True)
ax_var.set_ylabel("$\\langle \\left( x_i^t - \\langle x_i \\rangle \\right)^2\\rangle_{\\rm pop}$")

ax_var.grid()

ax_var.set_title("C",{'fontweight' : 'bold'}, loc="left")

fig_var.tight_layout()
fig_var.savefig("../plots/var.png", dpi=300)
#####################################

#====================================
dt_subsample = 1000

l_abs_max = np.ndarray((int(n_t_def/dt_subsample)))

for k in tqdm(range(int(n_t_def/dt_subsample))):
    l_abs_max[k] = np.abs(np.linalg.eigvals((DN.W.T*DN.gain_rec[k*dt_subsample,:]).T)).max()

fig_eig, ax_eig = plt.subplots(figsize=std_figsize, dpi=dpi_screen)

t_ax_eig = np.array(range(int(n_t_def/dt_subsample)))*dt_subsample

ax_eig.plot(t_ax_eig, np.log(l_abs_max))

ax_eig.set_xlabel("Time Steps")
ax_eig.set_ylabel("${\\rm log \\left(argmax\\{Re(\\lambda^t)\\}\\right) }$")

ax_eig.grid()

ax_eig.set_title("D",{'fontweight' : 'bold'}, loc="left")

fig_eig.tight_layout()
fig_eig.savefig("../plots/eig.png", dpi=300)
#####################################

#====================================
from PIL import Image

comp_size = (int(textwidth*300),int(6.*300.))

im_comp = Image.new('RGBA', comp_size, color=(255,255,255,255))

images = [Image.open("../plots/act_trans.png"),
        Image.open("../plots/gain.png"),
        Image.open("../plots/var.png"),
        Image.open("../plots/eig.png"),
        Image.open("../plots/act.png")]

images_scaled = []

scaled_img_width = int(comp_size[0]/2)

for img in images:

    img_new_size =  (scaled_img_width,int(img.size[1]*scaled_img_width/img.size[0]))
    images_scaled.append(img.resize(img_new_size,resample=Image.BICUBIC))

for k in range(2):
    for l in range(2):
        box = [0,0,0,0]
        box[0] = int(l*comp_size[0]/2)
        box[1] = int(k*images_scaled[2*k+l].size[1])
        box[2] = box[0] + int(comp_size[0]/2)
        box[3] = box[1] + int(images_scaled[2*k+l].size[1])

        im_comp.paste(images_scaled[2*k+l],box)

box = [0,0,0,0]
box[0] = 0
box[1] = int(2*images_scaled[4].size[1])
box[2] = int(comp_size[0]/2)
box[3] = box[1] + int(images_scaled[4].size[1])

im_comp.paste(images_scaled[4],box)

im_comp.save("../plots/im_comp.png")
#####################################

plt.show()


import pdb; pdb.set_trace()
