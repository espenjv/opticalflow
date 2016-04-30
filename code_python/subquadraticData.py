import HS_method as HS
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def subquadraticData(g,m,n,c,regu,diff_method):
    eps = 0.1

    # D-matrix from model term
    D = HS.makeDmatrix(g,m,n,diff_method)
    # Smoothness term
    V = HS.smoothnessHS(m,n)

    w = np.zeros(2*m*n)


    # Convex penaliser
    psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w)+ c,2) +math.pow(eps,2))),0)
    psi_d = sparse.kron(sparse.eye(2),psi_deriv)

    b = -psi_d.dot((D.T).dot(c))

    M = psi_d.dot((D.T).dot(D))

    del_w = 1

    iter_max = 50
    iter_nr = 0
    print np.max(del_w)
    while np.max(del_w) > math.pow(10,-6) and iter_nr < iter_max:
        print iter_nr
        iter_nr += 1
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)

        psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w_new)+ c,2) +math.pow(eps,2))),0)
        psi_d = sparse.kron(sparse.eye(2),psi_deriv)
    	b = -psi_d.dot((D.T).dot(c))

        M = psi_d.dot((D.T).dot(D))

        del_w = abs(w_new - w)

        w = w_new

    return w


def makeSubquadratic(w,m,n,c,D,eps):

    # Convex penaliser
    psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w)+ c,2) +math.pow(eps,2))),0)
    psi_d = sparse.kron(sparse.eye(2),psi_deriv)
    b = -psi_d.dot((D.T).dot(c))

    M = psi_d.dot((D.T).dot(D))

    return M,b
