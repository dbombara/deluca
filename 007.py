import numpy as np
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

A, B, dt = double_integrator_dt()
A_true, B_true, _ = double_integrator_dt(m=1.2, b=0.6, k=6)
Q, R = 500*jnp.eye(2), jnp.eye(1)
T = 200


# Square Wave
tv = np.arange(0,T)*dt
w1 = np.zeros_like(tv)
w21 = 100*dt*square(tv*0.3,duty=0.5)

# Constant Offset
w22 = 100*dt*np.ones_like(tv)

# Gaussian Noise
np.random.seed(seed=0)
w23 = 10*dt*np.random.normal(0,1.0,np.size(tv))

from matplotlib import rcParams as rcp
rcp['font.family'],rcp['font.serif'] = 'serif', ['Computer Modern Roman']
rcp['text.usetex'] = True
rcp['font.size'], rcp['axes.labelsize'],rcp['axes.titlesize'] = 20, 20,20
rcp['xtick.labelsize'],rcp['ytick.labelsize'] = 16, 16

fig, ax = plt.subplots(1,1,figsize=(10,4))
ax.plot(tv,w1, label="$w_{t \mid 1}$ (all profiles)",linewidth=3)
ax.plot(tv,w21,label="Square Wave, $w_{t \mid 2}$",linewidth=3)
ax.plot(tv,w22,label="Constant Bias, $w_{t \mid 2}$",linestyle="--",linewidth=3)
ax.plot(tv,w23,label="Gaussian, $w_{t \mid 2}$",linewidth=3)
ax.set_ylim(bottom=-12, top=30)
ax.set_title("Disturbance Profiles")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$w_t$")
ax.legend(ncol=2,loc="upper left")

plt.subplots_adjust(wspace=0.8, hspace=0.1)
plt.savefig("img/007.pdf",bbox_inches="tight")
plt.show()





