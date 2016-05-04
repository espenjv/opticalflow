import HS_method as HS
import DataTermMethods as dt
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


def subquadraticData(g1,g2,m,n,regu,diff_method,MotionTerm,gamma):

    g = g1
    eps = 0.1
    # Smoothness term
    V = HS.smoothnessHS(m,n)
    c = np.subtract(g2,g1)
    w = np.zeros(2*m*n)
    if MotionTerm is 'BCA':

        # D-matrix from model term
        D = HS.makeDmatrix(g,m,n,diff_method)
        # Convex penaliser
        psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w)+ c,2) +math.pow(eps,2))),0)
        psi_d = sparse.kron(sparse.eye(2),psi_deriv)

        b = -psi_d.dot((D.T).dot(c))

        M = psi_d.dot((D.T).dot(D))

        del_w = 1

        iter_max = 30
        iter_nr = 0
        while np.max(del_w) > math.pow(10,-6) and iter_nr < iter_max:
            print iter_nr
            iter_nr += 1
            G = M + math.pow(regu,-2)*V
            [G,b] = HS.neumann_boundary(G,b,m,n)
            w_new = spsolve(G,b)

            psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w_new)+ c,2) +math.pow(eps,2))),0)
            psi_d = sparse.kron(sparse.eye(2),psi_deriv)
            b = -psi_d.dot((D.T).dot(c))

            M = psi_d.dot((D.T).dot(D))

            del_w = abs(w_new - w)

            w = w_new

        return w
    if MotionTerm is 'GCA':
        [grad_g, grad_gx, grad_gy, gt, gxt, gyt] = dt.MotionTerms(g1,g2,m,n,diff_method)
        # Convex penaliser
        psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power((grad_g + gamma*(grad_gx + grad_gy)).dot(w)+ gt+gamma*(gxt+gyt),2) +math.pow(eps,2))),0)
        psi_d = sparse.kron(sparse.eye(2),psi_deriv)

        # # Model term
        M = psi_d.dot(((grad_g + grad_gx + grad_gy).T).dot(grad_g + gamma*(grad_gx + grad_gy)))
        # RHS
        b = - psi_d.dot(((grad_g + grad_gx + grad_gy).T).dot(gt + gamma*(gxt + gyt)))
        del_w = 1

        iter_max = 50
        iter_nr = 0
        while np.max(del_w) > math.pow(10,-6) and iter_nr < iter_max:
            print iter_nr
            iter_nr += 1
            G = M + math.pow(regu,-2)*V
            [G,b] = HS.neumann_boundary(G,b,m,n)
            w_new = spsolve(G,b)

            psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power((grad_g + gamma*(grad_gx + grad_gy)).dot(w)+ gt+gamma*(gxt+gyt),2) +math.pow(eps,2))),0)
            psi_d = sparse.kron(sparse.eye(2),psi_deriv)
            # # Model term
            M = psi_d.dot(((grad_g + grad_gx + grad_gy).T).dot(grad_g + gamma*(grad_gx + grad_gy)))
            # RHS
            b = - psi_d.dot(((grad_g + grad_gx + grad_gy).T).dot(gt + gamma*(gxt + gyt)))
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
