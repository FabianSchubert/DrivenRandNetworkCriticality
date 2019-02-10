#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from simulation import driven_net
from tqdm import tqdm

# Parameters
### Import standard parameters
from standard_params import *

std_in_def = .1
std_act_target_def = .2

n_t_def = 120000

DN = driven_net(N_net_def,
                cf_net_def,
                std_conn_def,
                std_in_def,
                mu_act_target_def,
                std_act_target_def,
                mu_bias_def,
                mu_gain_def,
                mu_trail_av_error_def,
                n_t_def,
                t_ext_off_def)

DN.run_sim()

DN.save_data("/media/fschubert/TOSHIBA EXT/simdata/")

'''
### Plotting

t_ax = np.array(range(n_t_def-100,n_t_def))

textwidth = 5.5532
std_figsize = (textwidth/2.,2.)
dpi_screen = 120

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
'''
