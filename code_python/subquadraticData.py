import HS_method as HS
import DataTermMethods as dtm
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


def subquadraticData(g1,g2,m,n,regu,diff_method,gamma,separate_pen,zeta,eps,normalize):
    g = g1
    # Smoothness term
    V = HS.smoothnessHS(m,n)
    w = np.zeros(2*m*n)
    del_w = 1
    iter_max = 20
    iter_nr = 0
    if separate_pen:
        while np.max(del_w) > math.pow(10,-2) and iter_nr < iter_max:
            print iter_nr
            M,b = dtm.makeSeparateSubquadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize)
            iter_nr += 1
            G = M + math.pow(regu,-2)*V
            [G,b] = HS.neumann_boundary(G,b,m,n)
            w_new = spsolve(G,b)
            del_w = abs(w_new - w)
            w = w_new
        return w
    else:
        while np.max(del_w) > math.pow(10,-2) and iter_nr < iter_max:
            print iter_nr
            M,b = dtm.makeSubquadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize)
            iter_nr += 1
            G = M + math.pow(regu,-2)*V
            [G,b] = HS.neumann_boundary(G,b,m,n)
            w_new = spsolve(G,b)
            del_w = abs(w_new - w)
            w = w_new
        return w
