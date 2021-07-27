"""
OCP for Two-body problem
"""


import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


g0 = 9.80665
isp = 3200.0
c1 = 0.014
c2 = 0.015
alf = 0.0
mu = 1.0


def ode_twobody(t, states, p):
	"""States containts position, velocity, mass, and their costates.
	ODE function must return array with shape (n,m), in the same layout as y
	"""
    tf = p[0]   # final time as free-parameter
    n, m = states.shape
    for j in range(m):
        state = states[:,j]
        # unpack state
        x  = state[0]
        y  = state[1]
        z  = state[2]
        vx = state[3]
        vy = state[4]
        vz = state[5]
        m  = state[6]
        # unpack costates as vectors
        lmbr = state[7:10]
        lmbv = state[10:13]
        lmbm = state[13]

        # compute norms
        r = state[0:3]
        r_norm = la.norm(r)
        lmbv_norm = la.norm(lmbv)

        # compute ustar
        ustar = min(max((c1/m * lmbv_norm + lmbm*c2 - alf)/(2*(1-alf)), 0), 1)

	    # # position, velocity, mass derivatives
	    # du[1:3] = r
	    # du[4:6] = -mu/r_norm^3 * u[1:3] - c1*ustar/m * lmbv/lmbv_norm
	    # du[7] = -c2*ustar
	    # compute co-states derivatives
	    du[8:10] = mu * lmbv/r_norm^3 - 3mu * np.dot(lmbv, r)*r/r_norm^5
	    du[11:13] = -lmbr
	    du[14] = -c1*ustar*lmbv_norm/m^2

        # initialize array for derivatives
        dstate = np.zeros(n,)
        # position derivatives
        dstate[0] = tf*( vx )
        dstate[1] = tf*( vy )
        dstate[2] = tf*( vz )
        # velocity derivatives
        dstate[3] = tf*( -mu/r_norm^3 * x - c1*ustar/m * lmbv[0]/lmbv_norm )
        dstate[4] = tf*( -mu/r_norm^3 * y - c1*ustar/m * lmbv[1]/lmbv_norm )
        dstate[5] = tf*( -mu/r_norm^3 * z - c1*ustar/m * lmbv[2]/lmbv_norm )
        # mass derivative
        dstate[6] = tf*( -c2*ustar )
        
        # derivatives of position costates
        dstate[7]  = tf*( mu * lmbv/r_norm^3 - 3mu * np.dot(lmbv, r)*r[0]/r_norm**5 )
        dstate[8]  = tf*( 0 )
        dstate[9]  = tf*( -p1 )
        # derivatives of velocity costates
        dstate[10] = tf*( -lmbr[0] )
        dstate[11] = tf*( -lmbr[1] )
        dstate[12] = tf*( -lmbr[2] )
        # derivative of mass costate
        dstate[13] = tf*( -c1*ustar*lmbv_norm/m**2 )
        
        # store
        if j == 0:
            dy = np.reshape(dstate, (n,1))
        else:
            dy = np.concatenate((dy, np.reshape(dstate, (n,1)) ), axis=1)
    return dy



def bc_transfer(ya, yb, p):
	"""Boundary conditions must have shape (n+k,), where ya & yb have shape (n,), and p has shape (k,)
	
	Args;
		ya (array): initial state
		yb (array): final state
		p (list): parameters
	"""
    res = np.zeros(len(ya)+len(p),)

     # ----- initial time boundary conditions ----- #
    res[0] = 1 - ya[0]       # x(0)  == 0
    res[1] = ya[1]       # y(0)  == 0
    res[2] = ya[2]       # z(0) == 0
    res[3] = ya[3]       # vx(0) == 0
    res[4] = 1.0 - ya[3] # vy(0) == 0
    res[5] = ya[3]       # vz(0) == 0
    res[6] = 1.0 - ya[3] # m(0) == 0
    
    # unpack final state
    xf  = yb[0]
    yf  = yb[1]
    zf  = yb[2]
    vxf = yb[3]
    vyf = yb[4]
    vzf = yb[5]
    mf  = yb[6]

    # unpack final costates
    lmbrf = yb[7:10]
    lmbvf = yb[10:13]
    lmbmf = yb[13]

    # compute norms
    rf = state[0:3]
    vf = state[3:6]
    rf_norm = la.norm(rf)
    lmbvf_norm = la.norm(lmbvf)

    # compute ustar
    ustar = min(max((c1/mf * lmbvf_norm + lmbmf*c2 - alf)/(2*(1-alf)), 0), 1)

    # fix final radius
    res[7] = rf_norm - 1.5

    # fix final hamiltonian to 0
    res[8] = np.dot(lmbrf,vf) - μ/rnorm**3*np.dot(lmbvf,rv) - c1*ustar/m*lmbvf_norm - lmbmf*c2*ustar + alf*ustar + (1-alf)*ustar**2

    # transversality conditions
    res[9] = lmbmf
    res[10] = np.dot(lmbrf, vf) - mu/rf_norm**3 * np.dot(lmbvf, rf)

	return res


