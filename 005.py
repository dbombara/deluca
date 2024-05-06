import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 18, 18,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

outfile = "data/005.npz"
nf = np.load(outfile)
learning_rates = nf["learning_rates"]
cost_lr = nf["cost_lr"]

fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3)) # Create a figure and axis for GPC
ax1.plot(learning_rates,cost_lr)
ax1.scatter(learning_rates,cost_lr)
ax1.set_title("Vary Learning Rate")
ax1.set_xscale('log')
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Cummulative Cost")
plt.subplots_adjust(wspace=0.8, hspace=0.1)
plt.savefig("img/005.pdf",bbox_inches="tight")
plt.show()