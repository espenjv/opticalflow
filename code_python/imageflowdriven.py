import NE_method as ID


import HS_method as HS
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def makeDiffusionMatrix(s1,s2,grad_w,m,n,eps):

    D1 = s1.dot(s1.T)
    D2 = s2.dot(s2.T)



    grad_u = grad_w[0:2*m*n]
    grad_v = grad_w[2*m*n:4*m*n]
    u_s1 = (s1.T).dot(grad_u)
    u_s2 = (s2.T).dot(grad_u)
    v_s1 = (s1.T).dot(grad_v)
    v_s2 = (s2.T).dot(grad_v)

    psiu1 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_s1,2)+math.pow(eps,2)))),0)
    psiu2 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_s2,2)+math.pow(eps,2)))),0)
    psiv1 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(v_s1,2)+math.pow(eps,2)))),0)
    psiv2 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(v_s2,2)+math.pow(eps,2)))),0)


    Diff_u = sparse.kron(sparse.eye(2),psiu1).dot(D1) + sparse.kron(sparse.eye(2),psiu2).dot(D2)


    Diff_v = sparse.kron(sparse.eye(2),psiv1).dot(D1) + sparse.kron(sparse.eye(2),psiv2).dot(D2)

    V = sparse.block_diag((Diff_u,Diff_v),format = "csr")

    return V


def findFlow(g,c,regu):
    eps = 0.8
    [m,n] = g.shape

    D = HS.makeDmatrix(g)
    L = HS.makeLmatrix(m,n)
    M = (D.T).dot(D)

    b = sparse.csr_matrix(-(D.T).dot(c))

    w = np.zeros(2*m*n)


    [s1,s2] = ID.dirDeriv(g)

    grad_w = L.dot(w)


    Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
    V = (L.T).dot(Dif.dot(L))

    del_w = 1



    while np.max(del_w) > math.pow(10,-2):
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)

        grad_w = L.dot(w_new)
        Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
        V = (L.T).dot(Dif.dot(L))

        del_w = abs(w_new - w)

        w = w_new

    return w
