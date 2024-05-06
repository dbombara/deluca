######## Vary the learning rate
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

from am232 import *

# Vary learning rate
cost_lr = np.array([])
learning_rates = np.array([0.000001, 0.00001, 0.00004, 0.00005, 0.00006, 0.00009, 0.0001, 0.0002, 0.0005, 0.001, 0.002])

for lr_test in learning_rates:
    A, B, dt = double_integrator_dt()
    A_true, B_true, _ = double_integrator_dt(m=1.2, b=0.6, k=6)
    Q, R = 500*jnp.eye(2), jnp.eye(1)

    # initialize classes
    agpc = GPC(
        A=A,
        B=B,
        Q = Q,
        R = R,
        H=4,
        HH=3,
        lr_scale=lr_test,
        decay= True,
        )
    alqr = LQR(A,B,Q,R)

    T = 200
    x0 = jnp.array([[0],[0]])
    tv = np.arange(0,T)*dt
    w2 = 100*dt*square(tv*0.3,duty=0.5)

    xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr = run_simulation(
        agpc, alqr, T, w2, x0, Q, R)
    cost_lr = np.append(cost_lr, np.sum(cv_gpc))
    
outfile = "data/005.npz"
np.savez(
    outfile,
    learning_rates = learning_rates,
    cost_lr = cost_lr,
    )