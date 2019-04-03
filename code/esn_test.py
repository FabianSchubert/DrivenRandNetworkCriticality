#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pdb

from esn_module import esn

from scipy.io.wavfile import read as readwav


#file = readwav("../data/bach.wav")

#pdb.set_trace()

#t = np.linspace(0.,100.,100)

#traindat = np.array([np.sin(t)]).T

#traindat = np.array([file[1][:300000,0]]).T

ESN = esn()

ESN.learn_data_prediction(traindat)

testdat = np.array([file[1][-300000:,0]]).T

pred_dat = ESN.predict_data(testdat)

pdb.set_trace()
