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

# vary b
cost_b = np.array([])
cost_b_lqr = np.array([])
b_vect = np.linspace(0.01,10,25)


for b_test in tqdm(b_vect):

    A, B, dt = double_integrator_dt(m=1, b=b_test, k=5)
    Q, R = 500*jnp.eye(2), jnp.eye(1)

    # initialize classes
    agpc = GPC(
        A=A,
        B=B,
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
    w2 = 100*dt*square(tv*0.3,duty=0.5) + 1

    xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr = run_simulation(agpc, alqr, T, w2, x0, Q, R)
    cost_b = np.append(cost_b, np.sum(cv_gpc))
    cost_b_lqr = np.append(cost_b_lqr, np.sum(cv_lqr))

outfile = "data/010.npz"

np.savez(
    outfile,
    tv = tv,
    xv_gpc = xv_gpc, 
    uv_gpc = uv_gpc, 
    wv_gpc = wv_gpc, 
    gv_gpc = gv_gpc, 
    cv_gpc = cv_gpc, 
    xv_lqr = xv_lqr, 
    uv_lqr = uv_lqr, 
    cv_lqr = cv_lqr,
    b_vect = b_vect,
    cost_b = cost_b,
    cost_b_lqr = cost_b_lqr,
)