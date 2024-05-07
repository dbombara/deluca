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

def double_integrator_dt(m=1,b=0.5,k=5,dt=0.1): 
    A = np.array([[0, 1], [-k/m, -b/m]])
    B = np.array([[0], [1/m]])
    n_states = A.shape[0]
    C, D= np.eye(1,n_states), 0
    sys = cont2discrete((A,B,C,D), dt, method='zoh', alpha=None)
    A,B = sys[0], sys[1]
    return A,B,dt

def quad_loss(x: jnp.ndarray, u: jnp.ndarray, Q, R):
    return (x.T @ Q @ x) + u.T @ R @ u

def run_simulation(agpc, alqr, T, w2, x0, Q, R):
    # initialize states
    agpc.state = x0
    x_gpc, x_lqr = x0, x0

    # intialize arrays of zeros
    uv_gpc, xv_gpc, wv_gpc, gv_gpc =  np.zeros((1,T)), np.zeros((2,T)), np.zeros((2,T)), np.zeros((1,T))
    cv_gpc = np.zeros((1,T))
    cv_lqr = np.zeros((1,T))
    uv_lqr, xv_lqr = np.zeros((1,T)), np.zeros((2,T))
    uv2_gpc = np.zeros((1,T))
    for i in tqdm(range(T)):
        #for i in range(T):

        # Compute disturbance 
        w = jnp.array([[0.],[w2[i]]])
        
        xv_gpc[:,i] = agpc.state.reshape((2,)) # put states into array
        xv_lqr[:,i] = x_lqr.reshape((2,)) # put the current state into a vector of all states
        
        # get control actions
        u_gpc, u2_gpc = agpc.get_action(agpc.state) # compute control action for GPC
        u_lqr = alqr(x_lqr) # compute the control action for the next iteration
        
        # compute cost
        c_gpc = quad_loss(x_gpc,u_gpc, Q, R) # compute cost for GPC
        c_lqr = quad_loss(x_lqr,u_lqr, Q, R) # compute cost for LQR
        
        # add noise and get next state
        
        x_gpc = agpc.A @ agpc.state + agpc.B @ u_gpc + w
        #w = (A_true @ agpc.state + B_true @ u_gpc) - (A @ agpc.state + B @ u_gpc) 
        agpc.update(x_gpc,u_gpc)
        
        #x_lqr = A_true @ x_lqr + B_true @ u_lqr # get the next state
        x_lqr = agpc.A @ x_lqr + agpc.B @ u_lqr + w
        
        # get policy loss
        g_gpc = agpc.policy_loss(agpc.M,agpc.noise_history)
        
        # put scalars into vectors
        uv_gpc[:,i], wv_gpc[:,i], gv_gpc[:,i] = u_gpc.reshape((1,)), w.reshape((2,)), g_gpc.reshape((1,))
        uv2_gpc[:,i] = u2_gpc.reshape((1,))
        cv_lqr[:,i] = c_lqr.reshape((1,))
        cv_gpc[:,i] = c_gpc.reshape((1,))
        uv_lqr[:,i] = u_lqr.reshape((1,)) 

    return xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr

def run_sim_modeling_error(agpc, alqr, T, w2, x0, Q, R):
    A, B, dt = double_integrator_dt(m=1,b=0.5,k=5)
    A_true, B_true, _ = double_integrator_dt(m=50, b=0, k=50)
    
    # initialize states
    agpc.state = x0
    x_gpc, x_lqr = x0, x0

    # intialize arrays of zeros
    uv_gpc, xv_gpc, wv_gpc, gv_gpc =  np.zeros((1,T)), np.zeros((2,T)), np.zeros((2,T)), np.zeros((1,T))
    cv_gpc = np.zeros((1,T))
    cv_lqr = np.zeros((1,T))
    uv_lqr, xv_lqr = np.zeros((1,T)), np.zeros((2,T))
    uv2_gpc = np.zeros((1,T))
    for i in tqdm(range(T)):
        #for i in range(T):

        # Compute disturbance 
        w = jnp.array([[0.],[w2[i]]])
        
        xv_gpc[:,i] = agpc.state.reshape((2,)) # put states into array
        xv_lqr[:,i] = x_lqr.reshape((2,)) # put the current state into a vector of all states
        
        # get control actions
        u_gpc, u2_gpc = agpc.get_action(agpc.state) # compute control action for GPC
        u_lqr = alqr(x_lqr) # compute the control action for the next iteration
        
        # compute cost
        c_gpc = quad_loss(x_gpc,u_gpc, Q, R) # compute cost for GPC
        c_lqr = quad_loss(x_lqr,u_lqr, Q, R) # compute cost for LQR
        
        # add noise and get next state
        
        #x_gpc = agpc.A @ agpc.state + agpc.B @ u_gpc + w
        x_gpc = A_true @ agpc.state + B_true @ u_gpc
        w = (A_true @ agpc.state + B_true @ u_gpc) - (A @ agpc.state + B @ u_gpc) 
        agpc.update(x_gpc,u_gpc)
        
        x_lqr = A_true @ x_lqr + B_true @ u_lqr # get the next state
        #x_lqr = agpc.A @ x_lqr + agpc.B @ u_lqr + w
        
        # get policy loss
        g_gpc = agpc.policy_loss(agpc.M,agpc.noise_history)
        
        # put scalars into vectors
        uv_gpc[:,i], wv_gpc[:,i], gv_gpc[:,i] = u_gpc.reshape((1,)), w.reshape((2,)), g_gpc.reshape((1,))
        uv2_gpc[:,i] = u2_gpc.reshape((1,))
        cv_lqr[:,i] = c_lqr.reshape((1,))
        cv_gpc[:,i] = c_gpc.reshape((1,))
        uv_lqr[:,i] = u_lqr.reshape((1,)) 

    return xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr

def run_sim_LTV(agpc: GPC, alqr: LQR, T, w2: float, x0, Q, R, mv=None, bv=None, kv=None):
    """Run simulation for LTV system

    Args:
        agpc (_type_): _description_
        alqr (_type_): _description_
        T (_type_): _description_
        w2 (_type_): _description_
        x0 (_type_): Initial state
        Q (_type_): State cost matrix
        R (_type_): Input cost matrix
        mv (_type_): Mass vector for each timestep
        bv (_type_): Vector of damping for each timestep
        kv (_type_): Vector of spring constant for each timestep
    """
    agpc.state, x_gpc, x_lqr = x0, x0, x0
    
    # intialize arrays of zeros
    uv_gpc, xv_gpc, wv_gpc, gv_gpc =  np.zeros((1,T)), np.zeros((2,T)), np.zeros((2,T)), np.zeros((1,T))
    cv_gpc = np.zeros((1,T))
    cv_lqr = np.zeros((1,T))
    uv_lqr, xv_lqr = np.zeros((1,T)), np.zeros((2,T))
    uv2_gpc = np.zeros((1,T))
    
    if mv is None:
        m = 1
        mv = m*np.ones((T,))
    if kv is None:
        k = 5
        kv = k*np.ones((T,))
    if bv is None:
        b = 0.5
        bv = b * np.ones((T,))
    
    for i in tqdm(range(T)):
        
        # Change system parameters
        A,B, _ = double_integrator_dt(m= mv[i], b=bv[i],k=kv[i])
        agpc.A, alqr.A, agpc.B, agpc.B = A, A, B, B
        K = LQR(A, B, Q, R).K
        alqr.K, agpc.K = K, K
        
        # Apply disturbance
        w = jnp.array([[0.],[w2[i]]])
        
        # Reshape state
        xv_gpc[:,i], xv_lqr[:,i] = agpc.state.reshape((2,)), x_lqr.reshape((2,))
        
        # Get control actions
        u_gpc, u2_gpc = agpc.get_action(agpc.state) # compute control action for GPC
        u_lqr = alqr(x_lqr) # compute the control action for the next iteration
        
        # compute cost
        c_gpc = quad_loss(x_gpc,u_gpc, Q, R) # compute cost for GPC
        c_lqr = quad_loss(x_lqr,u_lqr, Q, R) # compute cost for LQR
        
        # add noise and get next state
        x_gpc = agpc.A @ agpc.state + agpc.B @ u_gpc + w
        #w = (A_true @ agpc.state + B_true @ u_gpc) - (A @ agpc.state + B @ u_gpc) 
        agpc.update(x_gpc,u_gpc)
        
        #x_lqr = A_true @ x_lqr + B_true @ u_lqr # get the next state
        x_lqr = agpc.A @ x_lqr + agpc.B @ u_lqr + w
        
        # get policy loss
        g_gpc = agpc.policy_loss(agpc.M,agpc.noise_history)
        
        # put scalars into vectors
        uv_gpc[:,i], wv_gpc[:,i], gv_gpc[:,i] = u_gpc.reshape((1,)), w.reshape((2,)), g_gpc.reshape((1,))
        uv2_gpc[:,i] = u2_gpc.reshape((1,))
        cv_lqr[:,i] = c_lqr.reshape((1,))
        cv_gpc[:,i] = c_gpc.reshape((1,))
        uv_lqr[:,i] = u_lqr.reshape((1,))
        
        return xv_gpc, uv_gpc, wv_gpc, gv_gpc, cv_gpc, xv_lqr, uv_lqr, cv_lqr 
    
        
         