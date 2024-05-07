# double integrator with modeling error

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

A, B, dt = double_integrator_dt(m=1,b=0.5,k=5)
Q, R = 500*jnp.eye(2), jnp.eye(1)

# initialize classes
agpc = GPC(
    A = A,
    B = B,
    Q = Q,
    R = R,
    H=4,
    HH=3,
    lr_scale=0.00005,
    decay= True,
    )
alqr = LQR(A,B,Q,R)

T = 200
x0 = jnp.array([[0],[10]])
tv = np.arange(0,T)*dt
w2 = 100*dt*square(tv*0.3,duty=0.5)

mv = np.linspace(start=1,stop=10,num=T)

xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr =run_sim_modeling_error(agpc,alqr,T,w2,x0,Q,R)

outfile = "data/008.npz"
np.savez(
    outfile,
    tv = tv,
    xv_gpc = xv_gpc,
    xv_lqr = xv_lqr,
    cv_gpc = cv_gpc,
    cv_lqr = cv_lqr,
    uv_gpc = uv_gpc,
    uv_lqr = uv_lqr,
    wv_gpc = wv_gpc,
    )