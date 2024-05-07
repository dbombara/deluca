import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 18, 18,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

nf = np.load("data/002.npz")
tv = nf["tv"]
xv_gpc = nf["xv_gpc"]
xv_lqr = nf["xv_lqr"]
cv_gpc = nf["cv_gpc"]
cv_lqr = nf["cv_lqr"]

f_ig1, ax1 = plt.subplots(3, 1, figsize=(6, 8.5)) # Create a figure and axis for GPC
ax1[0].plot(tv,xv_gpc[0,:],label="$x_1$")
ax1[0].plot(tv,xv_gpc[1,:],label="$x_2$")
ax1[0].plot(tv, np.zeros_like(tv),label="$x_{1 \mid {ref}} = x_{2 \mid {ref}}$")
ax1[0].set_title("State (GPC)")
ax1[0].set_xlabel("Time (s)")
ax1[0].set_ylabel("State")
ax1[0].legend(loc="upper right", ncol=3, prop={'size': 14})
ax1[0].set_ylim(bottom=-2,top=15)

ax1[1].plot(tv,xv_lqr[0,:],label="$x_1$")
ax1[1].plot(tv,xv_lqr[1,:],label="$x_2$")
ax1[1].plot(tv, np.zeros_like(tv),label="$x_{1 \mid {ref}} = x_{2 \mid {ref}}$")
ax1[1].set_title("State (LQR)")
ax1[1].set_xlabel("Time (s)")
ax1[1].set_ylabel("State")
ax1[1].legend(loc="upper right", ncol=3, prop={'size': 14})
ax1[1].set_ylim(bottom=-2,top=15)

ax1[2].plot(tv, cv_gpc[0,:],label='GPC')
ax1[2].plot(tv, cv_lqr[0,:],label='LQR')
ax1[2].set_title("Cost")
ax1[2].set_xlabel("Time (s)")
ax1[2].set_ylabel("Cost")
ax1[2].legend(loc="upper right", ncol=2, prop={'size': 14})

plt.subplots_adjust(wspace=0.3, hspace=0.7)
plt.savefig("img/002.pdf",bbox_inches="tight")
plt.show()