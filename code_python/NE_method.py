import numpy as np
import HS_method as HS
from scipy import sparse
import math


def dirDeriv(g):
    k = 0.0001
    [Dx, Dy] = HS.imageDiff(g)
    grad = np.sqrt(np.power(Dx,2) + np.power(Dy,2))
    grad = grad + k
    sx = sparse.diags(np.divide(Dx,grad),0,format='csr')
    sy = sparse.diags(np.divide(Dy,grad),0,format='csr')
    s1 = sparse.hstack((sx,sy),format='csr').T
    s2 = sparse.hstack((-sy,sx),format='csr').T
    return s1,s2

def smoothnessNE(g,kappa):

    [m,n] = g.shape

    [Dx, Dy] = HS.imageDiff(g)
    grad = np.power(Dx,2) + np.power(Dy,2)
    denom = grad + 2*math.pow(kappa,2)
    sx = np.divide(np.power(Dx,2),denom)
    sy = np.divide(np.power(Dy,2),denom)
    sxy = np.divide(np.multiply(Dx,Dy),denom)

    P_diag = np.hstack([sx,sy])
    P1 = sparse.diags([P_diag+math.pow(kappa,2),-sxy,-sxy],[0,m*n,-m*n])
    P = sparse.kron(sparse.eye(2),P1,format = 'csr')

    L = HS.makeLmatrix(m,n)

    V = ((L.T).dot(P)).dot(L)
    return V
