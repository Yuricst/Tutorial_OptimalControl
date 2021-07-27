"""
Fixed time TPBVP for two-body problem

Ref: 
[1] L. Hou-Yuan and Z. Chang-Yin, 
    “Optimization of Low-Thrust Trajectories Using an Indirect Shooting Method without Guesses of Initial Costates,” 
    Chinese Astron. Astrophys., vol. 36, no. 4, pp. 389–398, 2012.

EoM has 7 states -> 14-dimension ODE
For fixed tf, we require 14 boundary conditions, given by:
    x(0)       = x_0   (6 components)
    x(tf)      = x_f   (6 components)
    m(0)       = m_0   (1 element)
    lmb_m(t_f) = 0     (1 element)
"""



import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


g0 = 9.80665
isp = 3200.0
tmax = 0.014
isp = 0.015
alf = 0.0
mu = 1.0

tf = 2*np.pi

m0 = 1.0
state_0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, m0])
state_f = np.array([1.52, 0.0, 0.0, 0.0, np.sqrt(1/1.52), 0.0])


def ode_twobody(t, states):
    """States containts position, velocity, mass, and their costates.
    ODE function must return array with shape (n,m), in the same layout as y
    """
    #tf = p[0]   # final time as free-parameter
    n, m = states.shape
    for j in range(m):
        state = states[:,j]
        # unpack state
        x,y,z,vx,vy,vz,m = state[0:7]
        # unpack costates as vectors
        lmbr = state[7:10]
        lmbv = state[10:13]
        lmbm = state[13]

        # compute norms
        r = state[0:3]
        r_norm = la.norm(r)
        lmbv_norm = la.norm(lmbv)

        # compute thrust direction
        ustar = -lmbv / lmbv_norm
        
        # compute thrust magnitude with switch function
        s = lmbv_norm/m + lmbm / (g0*isp)
        if s < 0.0:
            thrust = 0.0
        elif (0 <= s) and (s <= 2*tmax):
            thrust = s/2
        else:
            thrust = tmax

        # initialize array for derivatives
        dstate = np.zeros(n,)
        # position derivatives
        dstate[0] = vx
        dstate[1] = vy
        dstate[2] = vz
        # velocity derivatives
        dstate[3] = -mu/r_norm**3 * x + thrust/m * ustar[0]
        dstate[4] = -mu/r_norm**3 * y + thrust/m * ustar[1]
        dstate[5] = -mu/r_norm**3 * z + thrust/m * ustar[2]
        # mass derivative
        dstate[6] = -thrust / (g0*isp)
        
        # derivatives of position costates
        dstate[7]  = mu/r_norm**3*lmbv[0] - 3*mu*np.dot(lmbv, r)/r_norm**5 * r[0]
        dstate[8]  = mu/r_norm**3*lmbv[1] - 3*mu*np.dot(lmbv, r)/r_norm**5 * r[1]
        dstate[9]  = mu/r_norm**3*lmbv[2] - 3*mu*np.dot(lmbv, r)/r_norm**5 * r[2]
        # derivatives of velocity costates
        dstate[10] = -lmbr[0]
        dstate[11] = -lmbr[1]
        dstate[12] = -lmbr[2]
        # derivative of mass costate
        dstate[13] = -thrust*lmbv_norm/m**2
        
        # store
        if j == 0:
            dy = np.reshape(dstate, (n,1))
        else:
            dy = np.concatenate((dy, np.reshape(dstate, (n,1)) ), axis=1)
    return dy



def bc_transfer(ya, yb):
    """Boundary conditions must have shape (n+k,), where ya & yb have shape (n,), and p has shape (k,)
    
    Args:
        ya (array): initial state
        yb (array): final state
        p (list): parameters
    """
    res = np.zeros(len(ya),)  #+len(p),)

    # ----- initial time boundary conditions ----- #
    res[0] = state_0[0] - ya[0]
    res[1] = state_0[1] - ya[1]
    res[2] = state_0[2] - ya[2]
    res[3] = state_0[3] - ya[3]
    res[4] = state_0[4] - ya[4]
    res[5] = state_0[5] - ya[5]
    res[6] = state_0[6] - ya[6]

    # ----- initial time boundary conditions ----- #
    res[7]  = state_f[0] - yb[0]
    res[8]  = state_f[1] - yb[1]
    res[9]  = state_f[2] - yb[2]
    res[10] = state_f[3] - yb[3]
    res[11] = state_f[4] - yb[4]
    res[12] = state_f[5] - yb[5]
    # mass costate
    res[13] = yb[13]

    return res


if __name__=="__main__":
    # prepare time-domain mesh
    x = np.linspace(0, 1, 10)
    # prepare initial geuss
    y = np.random.randn(14, x.size) 

    # solve bvp
    sol = solve_bvp(fun=ode_twobody, bc=bc_transfer, x=x, y=y, verbose=2, bc_tol=1e-12)

    print(sol.status)
    print(sol.success)


