import NE_method as ID
import multigrid as mg
import DataTermMethods as dtm
import math
import HS_method as HS
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve,eigsh
# from multiprocessing import Pool


def makeDiffusionMatrix(r1,r2,grad_w,m,n,lam):
    # Forms the diffusion matrix in the lagged diffusivity iteration using a
    # non-Convex penaliser
    # Parameters: r1: eigenvector corresponding to the largest eigenvalue of
    #               the regularization matrix R
    #             r2: eignvector corresponding to the smallest eigenvalue of
    #               the regularization matrix R
    #            grad_w: flow gradient (derivatives)
    #            m,n: dimensions of image
    #             lam: regularization parameter
    # Returns:   Diff: Diffusion matrix
    eps = 0.001

    # Flow derivatives
    grad_u = grad_w[0:2*m*n]
    grad_v = grad_w[2*m*n:4*m*n]
    u_r1 = (r1.T).dot(grad_u)
    u_r2 = (r2.T).dot(grad_u)
    v_r1 = (r1.T).dot(grad_v)
    v_r2 = (r2.T).dot(grad_v)

    # Convex penaliser
    psi1 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_r1,2) + np.power(v_r1,2) + math.pow(eps,2)))),0)
    psi2 = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_r2,2) + np.power(v_r2,2) + math.pow(eps,2)))),0)


    # Flow-term function psi (non-Convex)
    # psi1 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(u_r1,2)+np.power(v_r1,2),math.pow(lam,2))),0)
    # psi2 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(u_r2,2)+np.power(v_r2,2),math.pow(lam,2))),0)

    D1 = (r1).dot(psi1.dot(r1.T))

    # Jointly penalise
    # D2 = r2.dot(psi2.dot(r2.T))

    # Single robust penalization
    D2 = r2.dot(r2.T)

    Diff = D1 + D2
    Diff = sparse.kron(sparse.eye(2),Diff)

    return Diff

def makeDiffusionMatrix2(s1,s2,grad_w,m,n,eps):
    eps = 0.001
    lam = 0.1
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

    # Non-convex
    # psiu1 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(u_s1,2),math.pow(lam,2))),0)
    # psiu2 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(u_s2,2),math.pow(lam,2))),0)
    # psiv1 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(v_s1,2),math.pow(lam,2))),0)
    # psiv2 = sparse.diags(np.divide(np.ones(m*n),np.ones(m*n) + np.divide(np.power(v_s2,2),math.pow(lam,2))),0)


    # Diffusion matrix "working on" the u-part
    Diff_u = sparse.kron(sparse.eye(2),psiu1).dot(D1) + sparse.kron(sparse.eye(2),psiu2).dot(D2)
    # Diffusion matrix for the v-part
    Diff_v = sparse.kron(sparse.eye(2),psiv1).dot(D1) + sparse.kron(sparse.eye(2),psiv2).dot(D2)

    V = sparse.block_diag((Diff_u,Diff_v),format = "csr")

    return V

def regTensorEigenvector(g,m,n,mu,zeta,gamma):
    # regTensorEigenvector: finds the eigenvectors of the regularization
    # matrix R (see framework in Optical Flow in Harmony)
    # Parameters: g: image as m-by-n 2-dimensional array
    # Returns: r1: eigenvector corresponding to the largest eigenvalue
    #          r2: eigenvector corresponding to the smallest eigenvalue

    # Parameter for second order derivatives
    k = 0.1 # Small parameter to avoid singular matrices
    [Dx, Dy] = HS.forwardDifferenceImage(g,m,n)
    [Dxx,Dxy] = HS.forwardDifferenceImage(Dx,m,n)
    [Dyx,Dyy] = HS.forwardDifferenceImage(Dy,m,n)

    # [Dxx,Dyy,Dxy] = HS.secondOrderDerivatives(g)
    # Dyx = Dxy

    # Normalisation terms
    theta_0 = np.power(Dx,2) + np.power(Dy,2) + zeta**2
    theta_x = np.power(Dxx,2) + np.power(Dxy,2) + zeta**2
    theta_y = np.power(Dyx,2) + np.power(Dyy,2) + zeta**2


    R11 = np.reshape(np.divide(np.power(Dx,2),theta_0) + np.divide(np.power(Dxx,2),theta_x) + np.divide(np.power(Dyx,2),theta_y),[n,m]).T
    R11 = ndimage.filters.gaussian_filter(R11,mu)
    R11 = np.reshape(R11.T,[1,m*n])[0]
    R12 = np.reshape(np.divide(np.multiply(Dx,Dy),theta_0) + np.divide(np.multiply(Dxx,Dxy),theta_x) + np.divide(np.multiply(Dyy,Dyx),theta_y),[n,m]).T
    R12 = ndimage.filters.gaussian_filter(R12,mu)
    R12 = np.reshape(R12.T,[1,m*n])[0]
    R21 = R12
    R22 = np.reshape(np.divide(np.power(Dy,2),theta_0) + np.divide(np.power(Dxy,2),theta_x) + np.divide(np.power(Dyx,2),theta_y),[n,m]).T
    R22 = ndimage.filters.gaussian_filter(R22,mu)
    R22 = np.reshape(R22.T,[1,m*n])[0]

    tmp = np.sqrt(np.power(R11,2)-2*np.multiply(R11,R22)+np.power(R22,2)+4*np.multiply(R21,R12))

    # r1 is the eigenvector corresponding to the largest eigenvalue
    r1_1 = R11 - R22 + tmp
    r1_2 = 2*R21
    norm1 = np.sqrt(np.power(r1_1,2)+np.power(r1_2,2)+ k)
    r1 = sparse.hstack((sparse.diags(np.divide(r1_1,norm1),0),sparse.diags(np.divide(r1_2,norm1),0)),format = 'csr').T

    r2_1 = R11 - R22 - tmp
    r2_2 = 2*R21
    norm2 = np.sqrt(np.power(r2_1,2)+np.power(r2_2,2) + k)
    r2 = sparse.hstack((sparse.diags(np.divide(r2_1,norm2),0),sparse.diags(np.divide(r2_2,norm2),0)),format = 'csr').T

    return r1,r2


def findFlow(g1,g2,m,n,regu,diff_method,mu_regTensor,data_penalize,zeta,eps,gamma,normalize):
    # Finds the flow vector for the method using the complementary
    # regularizer by the lagged diffusivity iteration scheme
    # Parameters: g: image
    #             c: time discretization
    #            regu: global scale regularization
    #       Diff_method: Method of discretization for the model term
    # Returns     w: flow vector
    g = g1
    # regularization parameter in the non-convex penaliser function
    lam = 0.1
    # Parameter in the lagged Diffusivity method
    t = 1
    L = HS.makeLmatrix(m,n)
    # RHS
    # Initial flow values
    w = np.ones(2*m*n)
    # Eigenvectors of the regularization methods
    [r1,r2] = regTensorEigenvector(g,m,n,mu_regTensor,zeta,gamma)

    del_w = 1
    iter_nr = 0
    iter_max = 20

    if data_penalize is 'quadratic':
        M,b = dtm.makeQuadraticDataTerm(g1,g2,m,n,diff_method,zeta,gamma,normalize)
        # Lagged Diffusivity iteration:
        while iter_nr <iter_max:
            print iter_nr
            iter_nr += 1
            # Flow derivatives
            grad_w = L.dot(w)
            # Diffusion matrix
            Dif = makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
            # Smoothness term, div(Dif*grad(w))
            V = (L.T).dot(Dif.dot(L))
            G = M + math.pow(regu,-2)*V
            w_new = spsolve(G,b)
            w_new = t*(w_new) + (t-1)*w
            del_w = abs(w_new - w)
            w = w_new
        return w
    elif data_penalize is 'subquadratic':
        if separate_pen:
            while iter_nr <iter_max:
                print iter_nr
                iter_nr += 1
                # Flow derivatives
                grad_w = L.dot(w)
                # Diffusion matrix
                Dif = makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
                # Smoothness term, div(Dif*grad(w))
                V = (L.T).dot(Dif.dot(L))
                M,b = dtm.makeSeparateSubquadraticDataTermQuadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize)
                G = M + math.pow(regu,-2)*V
                w_new = spsolve(G,b)
                w_new = t*(w_new) + (t-1)*w
                del_w = abs(w_new - w)
                w = w_new
            return w
        else:
            while iter_nr <iter_max:
                print iter_nr
                iter_nr += 1
                # Flow derivatives
                grad_w = L.dot(w)
                # Diffusion matrix
                Dif = makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
                # Smoothness term, div(Dif*grad(w))
                V = (L.T).dot(Dif.dot(L))
                M,b = dtm.makeSubquadraticDataTermQuadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize)
                G = M + math.pow(regu,-2)*V
                w_new = spsolve(G,b)
                w_new = t*(w_new) + (t-1)*w
                del_w = abs(w_new - w)
                w = w_new
            return w
