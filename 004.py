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
from am232 import *
from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 18, 18,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

outfile = "data/004.npz"
nf = np.load(outfile)
Hv = nf["Hv"]
cost_H = nf["cost_H"]


fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3)) # Create a figure and axis for GPC
ax1.plot(Hv,cost_H)
ax1.scatter(Hv,cost_H)
ax1.set_title("Varying Lookback Window $H$")
ax1.set_xlabel("Size of Lookback Window")
ax1.set_ylabel("Cummulative Cost")
plt.subplots_adjust(wspace=0.8, hspace=1)
plt.savefig("img/004.pdf",bbox_inches="tight")
plt.show()