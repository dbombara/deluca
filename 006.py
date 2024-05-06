# Vary the state cost

import deluca
import jax
import control as ct
import numpy as np
from scipy.signal import cont2discrete
from scipy.signal import square
import jax.numpy as jnp
from deluca.agents._gpc import GPC
from deluca.agents._lqr import LQR
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams

from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 18, 18,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

from am232 import *

nf = np.load("data/006.npz")
Q_vect = nf["Q_vect"]
cost_Q = nf["cost_Q"]
cost_Q_lqr = nf["cost_Q_lqr"]

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3)) # Create a figure and axis for GPC
ax1.plot(Q_vect,cost_Q,label='GPC')
ax1.plot(Q_vect,cost_Q_lqr,label='LQR')
ax1.scatter(Q_vect,cost_Q)
ax1.set_title("Vary State Cost")
ax1.set_xlabel("State Cost")
ax1.scatter(Q_vect,cost_Q_lqr)
ax1.set_ylabel("Cumulative Cost")
ax1.legend(loc="upper left", ncol=2)

plt.subplots_adjust(wspace=0.3, hspace=0.7)
plt.savefig("img/006.pdf",bbox_inches="tight")
plt.show()