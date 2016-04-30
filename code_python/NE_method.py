import numpy as np
import HS_method as HS
from scipy import sparse
import math


def dirDeriv(g,m,n):
    # Finds the unit vevtors in the direction normal to the image edges and
    # parallel to the image edges (see framework in Optical Flow in Harmony)
    # Parameters: g: image as m-by-n 2-dimensional array
    # Returns: s1: direction normal to image edges
    #          s2: direction parallel to image edges

    k = 0.0001 # Small parameter to avoid singular matrices
    [Dx, Dy] = HS.forwardDifferenceImage(g,m,n)
    grad = np.sqrt(np.power(Dx,2) + np.power(Dy,2) + np.power(k,2))
    sx = sparse.diags(np.divide(Dx,grad),0,format='csr')
    sy = sparse.diags(np.divide(Dy,grad),0,format='csr')
    s1 = sparse.hstack((sx,sy),format='csr').T
    s2 = sparse.hstack((-sy,sx),format='csr').T
    return s1,s2

def smoothnessNE(g,m,n,kappa):
    # Computes the anisotropic image driven smoothness term
    # of Nagel and Enkelmann.
    # Parameters: g: image as m-by-n array
    #          kappa: regularization parameter
    # Returns: V: smoothness array

    [Dx, Dy] = HS.forwardDifferenceImage(g,m,n)
    grad = np.power(Dx,2) + np.power(Dy,2) # The square of the gradient
    denom = grad + 2*math.pow(kappa,2) # Denominator in the NE diffusion matrix
    sx = np.divide(np.power(Dx,2),denom) # gx^2/denominator
    sy = np.divide(np.power(Dy,2),denom) # gy^2/denominator
    sxy = np.divide(np.multiply(Dx,Dy),denom) # gx gy/denominator

    P_diag = np.hstack([sx,sy]) # The main diagonal in the diffusion matrix
    P1 = sparse.diags([P_diag+math.pow(kappa,2),-sxy,-sxy],[0,m*n,-m*n])
    P = sparse.kron(sparse.eye(2),P1,format = 'csr') # Diffusion matrix

    L = HS.makeLmatrix(m,n)

    V = ((L.T).dot(P)).dot(L)
    return V
