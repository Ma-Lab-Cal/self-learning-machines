import PySpice.Spice.Xyce
import PySpice.Spice.Xyce.Server
from helpers import *
from spice_net import *

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse.linalg as spla
from numba import jit
import numba

# NMOS parameters
VTO = 0.0
KP = 2.0e-5
LIN_COEFF = 0.

# NMOS parameters
VTO = 0.0
KP = 2.0e-5
LIN_COEFF = 0.

# NMOS equations
@jit
def f(Vds, Vgs):
    if Vds >= 0: # forward active
        if Vgs - VTO < 0:
            return 0.
        if Vgs - VTO < Vds:
            return KP * (Vgs - VTO)**2 * (1 + LIN_COEFF * Vds)
        else:
            return KP * Vds * (2 * (Vgs - VTO) - Vds) * (1 + LIN_COEFF * Vds)
        
    else: # reverse active
        Vgd = Vgs - Vds
        if Vgd - VTO < 0:
            return 0.
        if Vgd - VTO < -Vds:
            return -KP * (Vgd - VTO)**2 * (1 - LIN_COEFF * Vds)
        else:
            return KP * Vds * (2 * (Vgd - VTO) + Vds) * (1 - LIN_COEFF * Vds)

@jit
def df(Vds, Vgs): # derivative wrt Vds 
    # TODO: should this be computed using autodiff?
    if Vds >= 0: # forward active
        if Vgs - VTO < 0:
            return 0.
        if Vgs - VTO < Vds:
            return KP * (Vgs - VTO)**2 * LIN_COEFF
        else:
            return -KP * (2 * (Vds - Vgs + VTO) + Vds * (3 * Vds - 4 * Vgs + 4 * VTO) * LIN_COEFF)
        
    else: # reverse active
        Vgd = Vgs - Vds
        if Vgd - VTO < 0:
            return 0.
        if Vgd - VTO < -Vds:
            return KP*LIN_COEFF*pow(-VTO - Vds + Vgs, 2) - KP*(-LIN_COEFF*Vds + 1)*(2*VTO + 2*Vds - 2*Vgs)
        else:
            return -KP*LIN_COEFF*Vds*(-2*VTO - Vds + 2*Vgs) - KP*Vds*(-LIN_COEFF*Vds + 1) + KP*(-LIN_COEFF*Vds + 1)*(-2*VTO - Vds + 2*Vgs)

@jit 
def op_newton(data, sp_indices, rhs, stamp_indices, params, operating_points):
    # zero out things that will be overwritten
    for (n1, n2) in stamp_indices:
        # M[n1, n1] = 1e-12
        # M[n1, n2] = 0
        # M[n2, n1] = 0
        # M[n2, n2] = 1e-12
        data[sp_indices[(n1, n1)]] = 1e-16
        data[sp_indices[(n1, n2)]] = 0
        data[sp_indices[(n2, n1)]] = 0
        data[sp_indices[(n2, n2)]] = 1e-16
        
    op_guessses = np.zeros_like(params)
    for i, (n1, n2) in enumerate(stamp_indices):
        Vds = operating_points[n1] - operating_points[n2]
        Vgs = params[i]

        geq = df(Vds, Vgs)
        # geq = df(Vds, Vgs)
        if abs(geq) < 1e-12:
            geq = 1e-12 * np.sign(geq)
        op_guessses[i] = geq

        # stamp nodal matrix
        # M[n1, n1] += geq
        # M[n1, n2] -= geq
        # M[n2, n1] -= geq
        # M[n2, n2] += geq
        data[sp_indices[(n1, n1)]] += geq
        data[sp_indices[(n1, n2)]] -= geq
        data[sp_indices[(n2, n1)]] -= geq
        data[sp_indices[(n2, n2)]] += geq

        # stamp rhs

        rhs[n1] += (f(Vds, Vgs) - geq * Vds)
        rhs[n2] -= (f(Vds, Vgs) - geq * Vds)
    
    # return M, rhs, op_guessses
    return data, rhs

def set_voltages(rhs, n, vals):
    rhs *= 0
    for i, v in enumerate(vals):
        rhs[n + i] = v
    return rhs

def check_convergence(vprev, vcurr, icurr, iprev, stamp_indices, ABSTOL, RELTOL, VNTOL):
    v_conv = np.all(np.abs(vprev - vcurr) < VNTOL + RELTOL * np.maximum(np.abs(vcurr), np.abs(vprev)))
    # SPICE computes the current using nonlinear functions of the previous iteration's voltages
    # not sure how we could do that
    i_conv = np.all(np.abs(iprev - icurr) < ABSTOL + RELTOL * np.maximum(np.abs(icurr), np.abs(iprev)))

    return v_conv and i_conv