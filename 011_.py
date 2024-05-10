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

# vary k
cost_k = np.array([])
cost_k_lqr = np.array([])
k_vect = np.linspace(5,500,25)


for k_test in tqdm(k_vect):

    A, B, dt = double_integrator_dt(m=1, b=0.5, k=k_test)
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

    xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr = run_simulation(agpc, alqr, T, w2, x0, Q,R)
    cost_k = np.append(cost_k, np.sum(cv_gpc))
    cost_k_lqr = np.append(cost_k_lqr, np.sum(cv_lqr))
    
    outfile = "data/011.npz"
    
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
        cost_k = cost_k,
        cost_k_lqr = cost_k_lqr,
        k_vect = k_vect,
    )
