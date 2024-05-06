import numpy as np
from scipy.signal import square
import jax.numpy as jnp
from deluca.agents._gpc import GPC
from deluca.agents._lqr import LQR
from am232 import *

"""Square Wave
Experiments for square wave
"""

# Square Wave Noise
A, B, dt = double_integrator_dt()
A_true, B_true, _ = double_integrator_dt(m=1.2, b=0.6, k=6)
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
x0 = jnp.array([[0],[0]])
tv = np.arange(0,T)*dt
w2 = 100*dt*square(tv*0.3,duty=0.5)

xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr = run_simulation(
    agpc, alqr, T, w2, x0, Q, R)

outfile = "data/001.npz"
np.savez(
    outfile,
    tv = tv,
    xv_gpc = xv_gpc,
    xv_lqr = xv_lqr,
    cv_gpc = cv_gpc,
    cv_lqr = cv_lqr
    )

