import HS_method as HS

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def findFlow(g,c,regu):
    eps = 1
    [m,n] = g.shape

    D = HS.makeDmatrix(g)
    L = HS.makeLmatrix(m,n)
    M = (D.T).dot(D)

    I = sparse.eye(m,n)
    I = sparse.hstack((I,I,I,I))

    w = np.zeros(2*m*n)

    psi_deriv = sparse.diags(np.divide(np.ones(m*n),(2*np.sqrt(I.dot(np.power(L.dot(w),2))+eps^2))))
    V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

    del_w = 1

    b = sparse.csr_matrix(-(D.T).dot(c))

    while max(del_u) > 10^(-2):
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)

        psi_deriv = sparse.diags(np.divide(np.ones(m*n),(2*np.sqrt(I.dot(np.power(L.dot(w_new),2))+eps^2))))
        V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

        del_w = abs(w_new - w)

        w = w_new

    return w
