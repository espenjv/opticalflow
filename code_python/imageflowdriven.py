import NE_method as ID
import subquadraticData as sq
import HS_method as HS
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

def makeDiffusionMatrix(s1,s2,grad_w,m,n,eps):
    # Forms the diffusion matrix in the lagged diffusivity iteration
    # Parameters: s1: unit vector normal to image edges
    #             s2: unit vector parallel to image edges
    #            grad_w: flow gradient (derivatives)
    #            m,n: dimensions of image
    #             eps: parameter in the convex penaliser function
    # Returns:   V: Diffusion matrix

    D1 = s1.dot(s1.T)
    D2 = s2.dot(s2.T)

    # Flow derivatives
    grad_u = grad_w[0:2*m*n]
    grad_v = grad_w[2*m*n:4*m*n]
    u_s1 = (s1.T).dot(grad_u)
    u_s2 = (s2.T).dot(grad_u)
    v_s1 = (s1.T).dot(grad_v)
    v_s2 = (s2.T).dot(grad_v)

    # Convex penaliser terms
    psiu1 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_s1,2)+math.pow(eps,2)))),0)
    psiu2 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_s2,2)+math.pow(eps,2)))),0)
    psiv1 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(v_s1,2)+math.pow(eps,2)))),0)
    psiv2 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(v_s2,2)+math.pow(eps,2)))),0)

    # Diffusion matrix "working on" the u-part
    Diff_u = sparse.kron(sparse.eye(2),psiu1).dot(D1) + sparse.kron(sparse.eye(2),psiu2).dot(D2)
    # Diffusion matrix for the v-part
    Diff_v = sparse.kron(sparse.eye(2),psiv1).dot(D1) + sparse.kron(sparse.eye(2),psiv2).dot(D2)

    V = sparse.block_diag((Diff_u,Diff_v),format = "csr")

    return V

def findFlow(g,m,n,c,regu,Diff_method,data_penalize,sigma_regTensor):
    # Finds the flow vector for the image and flow driven method using lagged
    # diffusivity iteration scheme
    # Parameters: g: image
    #             c: time discretization
    #            regu: global scale regularization
    # Returns     w: flow vector

    if data_penalize is 'quadratic':

        # regularization parameter in the convex penaliser function (flow driven)
        eps = 0.001
        D = HS.makeDmatrix(g,m,n,Diff_method)
        L = HS.makeLmatrix(m,n)
        M = (D.T).dot(D)
        # RHS
        b = -(D.T).dot(c)
        # Initial flow values
        w = np.zeros(2*m*n)

        # Directions nomal to edges and parallel to edges
        [s1,s2,trace] = ID.dirDeriv(g,m,n,sigma_regTensor)
        # Flow derivatives
        grad_w = L.dot(w)
        # Diffusion matrix
        Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
        # Smoothness term, div(Dif*grad(w))
        V = (L.T).dot(Dif.dot(L))

        del_w = 1
        iter_nr = 0
        iter_max = 20
        # Lagged Diffusivity iteration:
        while np.max(del_w) > math.pow(10,-4) and iter_nr <iter_max:
            print iter_nr
            iter_nr += 1
            G = M + math.pow(regu,-2)*V
            [G,b] = HS.neumann_boundary(G,b,m,n)
            w_new = spsolve(G,b)

            grad_w = L.dot(w_new)
            Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
            V = (L.T).dot(Dif.dot(L))

            del_w = abs(w_new - w)

            w = w_new

        return w

    elif data_penalize is 'subquadratic':
            # regularization parameter in the convex penaliser function (flow driven)
            eps = 0.001
            D = HS.makeDmatrix(g,m,n,Diff_method)
            L = HS.makeLmatrix(m,n)
            # Initial flow values
            w = np.zeros(2*m*n)

            [M,b] = sq.makeSubquadratic(w,m,n,c,D,eps)

            # Directions nomal to edges and parallel to edges
            [s1,s2,trace] = ID.dirDeriv(g,m,n,sigma_regTensor)
            # Flow derivatives
            grad_w = L.dot(w)
            # Diffusion matrix
            Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
            # Smoothness term, div(Dif*grad(w))
            V = (L.T).dot(Dif.dot(L))

            del_w = 1
            iter_nr = 0
            iter_max = 20
            # Lagged Diffusivity iteration:
            while np.max(del_w) > math.pow(10,-4) and iter_nr <iter_max:
                print iter_nr
                iter_nr += 1
                G = M + math.pow(regu,-2)*V
                w_new = spsolve(G,b)
                [M,b] = sq.makeSubquadratic(w_new,m,n,c,D,eps)
                grad_w = L.dot(w_new)
                Dif = makeDiffusionMatrix(s1,s2,grad_w,m,n,eps)
                V = (L.T).dot(Dif.dot(L))

                del_w = abs(w_new - w)

                w = w_new

            return w
