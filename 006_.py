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

# Vary State Cost
cost_Q = np.array([])
cost_Q_lqr = np.array([])
Q_vect = np.linspace(100,1000,10)
lr_test = 0.002

for Q_test in Q_vect:
    A, B, dt = double_integrator_dt()
    A_true, B_true, _ = double_integrator_dt(m=1.2, b=0.6, k=6)
    Q, R = Q_test*jnp.eye(2), jnp.eye(1)
    # initialize classes
    agpc = GPC(A=A,B=B,Q = Q,R = R,H=4,HH=3,lr_scale=lr_test,decay= True,)
    alqr = LQR(A,B,Q,R)

    T = 200
    x0 = jnp.array([[0],[0]])
    tv = np.arange(0,T)*dt
    w2 = 100*dt*square(tv*0.3,duty=0.5) + 1

    xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr = run_simulation(agpc, alqr, T, w2, x0, Q, R)
    cost_Q = np.append(cost_Q, np.sum(cv_gpc))
    cost_Q_lqr = np.append(cost_Q_lqr, np.sum(cv_lqr))

outfile = "data/006.npz"
np.savez(
    outfile,
    Q_vect = Q_vect,
    cost_Q = cost_Q,
    cost_Q_lqr = cost_Q_lqr,
)