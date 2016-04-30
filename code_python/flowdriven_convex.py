import HS_method as HS

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def findFlow(g,c,regu,Diff_method):
    # Computes the flow using lagged diffusivity
    # Parameters: g: image
    #             c: time discretization
    #            regu: global scale regularization
    # Returns     w: flow vector

    # Regularization parameter in the convex penaliser
    eps = 0.0001
    [m,n] = g.shape

    # D, L and Model term
    D = HS.makeDmatrix(g,Diff_method)
    L = HS.makeLmatrix(m,n)
    M = (D.T).dot(D)

    I = sparse.eye(m*n)
    I = sparse.hstack((I,I,I,I))
    b = -(D.T).dot(c)

    w = np.zeros(2*m*n)
    # Flow derivatives
    grad_w = L.dot(w)
    # Derivatives of the components
    u_x = grad_w[0:m*n]
    u_y = grad_w[m*n:2*m*n]
    v_x = grad_w[2*m*n:3*m*n]
    v_y = grad_w[3*m*n:4*m*n]

    # Convex penaliser
    psi_deriv = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_x,2) + np.power(u_y,2) + np.power(v_x,2) + np.power(v_y,2)+math.pow(eps,2)))),0)
    # Div(Dif grad(w))
    V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

    del_w = 1

    iter_max = 20
    iter_nr = 0
    print np.max(del_w)
    while np.max(del_w) > math.pow(10,-6) and iter_nr < iter_max:
        print iter_nr
        iter_nr += 1
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)
        grad_w = L.dot(w_new)

        u_x = grad_w[0:m*n]
        u_y = grad_w[m*n:2*m*n]
        v_x = grad_w[2*m*n:3*m*n]
        v_y = grad_w[3*m*n:4*m*n]


        psi_deriv = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_x,2) + np.power(u_y,2) + np.power(v_x,2) + np.power(v_y,2)+math.pow(eps,2)))),0)
        V = ((L.T).dot(sparse.kron(sparse.eye(4),psi_deriv,format = 'csr'))).dot(L)

        del_w = abs(w_new - w)

        w = w_new

    print iter_nr

    return w
