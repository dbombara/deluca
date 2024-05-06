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
from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 18, 18,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16
#######################

nf = np.load("data/001.npz")
tv = nf["tv"]
xv_gpc = nf["xv_gpc"]
xv_lqr = nf["xv_lqr"]
cv_gpc = nf["cv_gpc"]
cv_lqr = nf["cv_lqr"]



#######################
fig1, ax1 = plt.subplots(2, 1,  figsize=(6, 6)) # Create a figure and axis for GPC
ax1[0].plot(tv,xv_gpc[0,:],label="$x_1$")
ax1[0].plot(tv,xv_gpc[1,:],label="$x_2$")
ax1[0].plot(tv, np.zeros_like(tv),label="$x_{ref} $")
ax1[0].set_title("State (GPC)")
ax1[0].set_xlabel("Time (s)")
ax1[0].set_ylabel("State")
ax1[0].legend(loc="upper right", ncol=3)
ax1[0].set_ylim(bottom=-18,top=30)

ax1[1].plot(tv,xv_lqr[0,:],label="$x_1$")
ax1[1].plot(tv,xv_lqr[1,:],label="$x_2$")
ax1[1].plot(tv, np.zeros_like(tv),label="$x_{ref}$")
ax1[1].set_title("State (LQR)")
ax1[1].set_xlabel("Time (s)")
ax1[1].set_ylabel("State")
ax1[1].legend(loc="upper right", ncol=3)
ax1[1].set_ylim(bottom=-18,top=30)
plt.subplots_adjust(wspace=0.3, hspace=0.7)
plt.savefig("img/001a.pdf",bbox_inches="tight")

fix,ax = plt.subplots(1,1,figsize=(6,3))
ax.plot(tv, cv_gpc[0,:],label='GPC')
ax.plot(tv, cv_lqr[0,:],label='LQR')
ax.set_title("Cost")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cost")
ax.legend(loc="upper left", ncol=1)

plt.subplots_adjust(wspace=0.3, hspace=0.7)
plt.savefig("img/001b.pdf",bbox_inches="tight")
plt.show()