import numpy as np
import HS_method as HS
from scipy import sparse

def smoothnessNE(g,kappa):

    [Dx, Dy] = HS.imageDiff(g)
    grad = np.power(Dx,2) + np.power(Dy,2)
    denom = grad + 2*kappa^2
    sx = np.divide(np.power(Dx,2),denom)
    sy = np.divide(np.power(Dy,2),denom)
    sxy = np.divide(np.multiply(Dx,Dy),denom)

    P_diag = np.hstack([sx,sy])
    P1 = sparse.diags([P_diag+kappa^2,-sxy,-sxy],[0,m*n,-m*n])
    P = sparse.kron(sparse.eye(2),P1,format = 'csr')

    L = HS.makeLmatrix(m,n)

    V = ((L.T).dot(P)).dot(L)
    return V
