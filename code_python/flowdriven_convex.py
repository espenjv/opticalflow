import HS_method as HS

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def findFlow(g,c,regu):
    eps = 0.8
    [m,n] = g.shape

    D = HS.makeDmatrix(g)
    L = HS.makeLmatrix(m,n)
    M = (D.T).dot(D)

    I = sparse.eye(m*n)
    I = sparse.hstack((I,I,I,I))
    b = sparse.csr_matrix(-(D.T).dot(c))

    w = np.zeros(2*m*n)


    psi_deriv = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(I.dot(np.power(L.dot(w),2))+math.pow(eps,2)))),0)
    V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

    del_w = 1


    while np.max(del_w) > math.pow(10,-2):
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)

        psi_deriv = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(I.dot(np.power(L.dot(w_new),2))+math.pow(eps,2)))),0)
        V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

        del_w = abs(w_new - w)

        w = w_new

    return w
