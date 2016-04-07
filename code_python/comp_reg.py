import NE_method as ID
import multigrid as mg

import math
import HS_method as HS
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve,eigsh

def makeDiffusionMatrix(r1,r2,grad_w,m,n,lam):
    eps = 0.0001
    # Forms the diffusion matrix in the lagged diffusivity iteration
    # Parameters: s1: unit vector normal to image edges
    #             s2: unit vector parallel to image edges
    #            grad_w: flow gradient (derivatives)
    #            m,n: dimensions of image
    #             eps: parameter in the convex penaliser function
    # Returns:   V: Diffusion matrix
    # Flow derivatives
    grad_u = grad_w[0:2*m*n]
    grad_v = grad_w[2*m*n:4*m*n]
    u_r1 = (r1.T).dot(grad_u)
    u_r2 = (r2.T).dot(grad_u)
    v_r1 = (r1.T).dot(grad_v)
    v_r2 = (r2.T).dot(grad_v)

    psi1 = sparse.diags(np.ones(m*n) + np.divide(np.power(u_r1,2)+np.power(v_r1,2),lam),0)
    psi2 = sparse.diags(np.ones(m*n) + np.divide(np.power(u_r2,2)+np.power(v_r2,2),lam),0)
    #psi = sparse.diags(np.divide(np.ones(m*n),(np.sqrt(np.power(u_r1,2) + np.power(v_r1,2)+math.pow(eps,2)))),0)
    D1 = (r1).dot(psi1.dot(r1.T))
    D2 = r2.dot(psi2.dot(r2.T))

    # non-Convex penaliser terms
    Diff = D1 + D2
    Diff = sparse.kron(sparse.eye(2),Diff)

    return Diff

def makeDiffusionMatrix2(s1,s2,grad_w,m,n,eps):
    eps = 0.001
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

def regTensorEigenvector(g):
    # parallel to the image edges (see framework in Optical Flow in Harmony)
    # Parameters: g: image as m-by-n 2-dimensional array
    # Returns: s1: direction normal to image edges
    #          s2: direction parallel to image edges

    gamma = 50
    k = 0.0001 # Small parameter to avoid singular matrices
    [m,n] = g.shape
    [Dx, Dy] = HS.forwardDifferenceImage(g)
    gx = np.reshape(Dx,[n,m]).T
    gy = np.reshape(Dy,[n,m]).T
    [Dxx,Dxy] = HS.forwardDifferenceImage(gx)


    [Dyx,Dyy] = HS.forwardDifferenceImage(gy)

    theta_0 = np.sqrt(np.power(Dx,2) + np.power(Dy,2))
    theta_x = sparse.diags(np.sqrt(np.power(Dxx,2) + np.power(Dxy,2)),0,format = 'csr')
    theta_y = sparse.diags(np.sqrt(np.power(Dyx,2) + np.power(Dyy,2)),0,format = 'csr')


    sx = sparse.diags(np.divide(Dx,theta_0),0,format='csr')
    sy = sparse.diags(np.divide(Dy,theta_0),0,format='csr')
    d = sparse.hstack((sx,sy),format='csr').T

    sxx = sparse.diags(Dxx,0,format='csr')
    sxy = sparse.diags(Dxy,0,format='csr')
    dx = sparse.hstack((sxx,sxy),format='csr').T

    syx = sparse.diags(Dyx,0,format='csr')
    syy = sparse.diags(Dyy,0,format='csr')
    dy = sparse.hstack((syx,syy),format='csr').T

    R = d.dot(d.T) + gamma*(dx.dot(dx.T)+ dy.dot(dy.T))
    R11 = R[0:m*n,0:m*n]
    R12 = R[0:m*n,m*n:2*m*n]
    R21 = R[m*n:2*m*n,0:m*n]
    R22 = R[m*n:2*m*n,m*n:2*m*n]


    tmp = np.sqrt(np.power(R11,2)-2*R11.dot(R22)+np.power(R22,2)+4*R21.dot(R21))

    # r1 is the eigenvector corresponding to the largest eigenvalue
    r1_1 = R11 - R22 + tmp
    r1_2 = 2*R21
    norm1 = np.sqrt(np.power(r1_1,2)+np.power(r1_2,2))
    r1 = sparse.hstack((np.divide(r1_1,norm1),np.divide(r1_2,norm1)),format = 'csr').T

    r2_1 = R11 - R22 - tmp
    r2_2 = 2*R21

    norm2 = np.sqrt(np.power(r2_1,2)+np.power(r2_2,2))
    r2 = sparse.hstack((np.divide(r2_1,norm2),np.divide(r2_2,norm2)),format = 'csr').T

    [w,v] = eigsh(R[[[0],[m*n]],[0,m*n]],1)


    return r1,r2


def findFlow(g,c,regu):
    # Finds the flow vector for the image and flow driven method using lagged
    # diffusivity iteration scheme
    # Parameters: g: image
    #             c: time discretization
    #            regu: global scale regularization
    # Returns     w: flow vector

    # regularization parameter in the convex penaliser function (flow driven)
    lam = 0.0001
    [m,n] = g.shape
    D = HS.makeDmatrix(g)
    L = HS.makeLmatrix(m,n)
    M = (D.T).dot(D)
    # RHS
    b = -(D.T).dot(c)
    # Initial flow values
    w = np.ones(2*m*n)

    # Directions nomal to edges and parallel to edges
    [r1,r2] = regTensorEigenvector(g)


    # Flow derivatives
    grad_w = L.dot(w)
    # Diffusion matrix
    Dif = makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
    # Smoothness term, div(Dif*grad(w))
    V = (L.T).dot(Dif.dot(L))

    del_w = 1
    iter_nr = 0
    iter_max = 50
    # Lagged Diffusivity iteration:
    while np.max(del_w) > math.pow(10,-2) and iter_nr <iter_max:
        print iter_nr
        iter_nr += 1
        G = M + math.pow(regu,-2)*V
        w_new = spsolve(G,b)

        grad_w = L.dot(w_new)
        Dif = makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
        V = (L.T).dot(Dif.dot(L))

        del_w = abs(w_new - w)

        print np.max(del_w)
        w = w_new


    return w
