import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 16, 14,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

nf = np.load("data/010.npz")
b_vect = nf["b_vect"]
cost_b = nf["cost_b"]
cost_b_lqr = nf["cost_b_lqr"]

fig1, ax1 = plt.subplots(3, 1, figsize=(5, 6)) # Create a figure and axis for GPC
ax1[0].plot(b_vect,cost_b,label='GPC',marker='o')
ax1[0].set_title("Vary Damping")
ax1[0].set_xlabel("Damping (Ns/m)")
ax1[0].set_ylabel("Total Cost")
ax1[0].legend(loc="upper left")

ax1[1].plot(b_vect,cost_b_lqr,label='LQR',marker='o',color='r')
ax1[1].set_xlabel("Damping (Ns/m)")
ax1[1].set_ylabel("Total Cost")
ax1[1].legend(loc="upper right")

ax1[2].plot(b_vect,(cost_b - cost_b_lqr),label='$\sum c_{GPC} - \sum c_{LQR}$',marker='o',color='k')
ax1[2].set_xlabel("Damping (Ns/m)")
ax1[2].set_title("$\sum c_{GPC} - \sum c_{LQR}$")
ax1[2].set_ylabel("Cost Diff.")
#ax1[2].legend(loc="upper left")

plt.subplots_adjust(wspace=0.3, hspace=1)
plt.savefig("img/010.pdf",bbox_inches="tight")

plt.show()