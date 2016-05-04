import numpy as np
import HS_method as HS
from scipy import sparse, ndimage
import math


def dirDeriv(g,m,n,sigma):
    # Finds the unit vevtors in the direction normal to the image edges and
    # parallel to the image edges (see framework in Optical Flow in Harmony)
    # Parameters: g: image as m-by-n 2-dimensional array
    # Returns: s1: direction normal to image edges
    #          s2: direction parallel to image edges

    k = 0.000001 # Small parameter to avoid singular matrices
    [Dx, Dy] = HS.forwardDifferenceImage(g,m,n)
    # grad = np.sqrt(np.power(Dx,2) + np.power(Dy,2) + np.power(k,2))
    # sx = sparse.diags(np.divide(Dx,grad),0,format='csr')
    # sy = sparse.diags(np.divide(Dy,grad),0,format='csr')
    # s1 = sparse.hstack((sx,sy),format='csr').T
    # s2 = sparse.hstack((-sy,sx),format='csr').T

    theta_0 = np.power(Dx,2) + np.power(Dy,2) + k


    S11 = np.reshape(np.divide(np.power(Dx,2),theta_0),[n,m]).T
    S11 = ndimage.filters.gaussian_filter(S11,sigma)
    S11 = np.reshape(S11.T,[1,m*n])[0]
    S12 = np.reshape(np.divide(np.multiply(Dx,Dy),theta_0),[n,m]).T
    S12 = ndimage.filters.gaussian_filter(S12,sigma)
    S12 = np.reshape(S12.T,[1,m*n])[0]
    S21 = S12
    S22 = np.reshape(np.divide(np.power(Dy,2),theta_0),[n,m]).T
    S22 = ndimage.filters.gaussian_filter(S22,sigma)
    S22 = np.reshape(S22.T,[1,m*n])[0]

    tmp = np.sqrt(np.power(S11,2)-2*np.multiply(S11,S22)+np.power(S22,2)+4*np.multiply(S21,S12))

    # s1 is the eigenvector corresponding to the largest eigenvalue
    s1_1 = S11 - S22 + tmp
    s1_2 = 2*S21
    norm1 = np.sqrt(np.power(s1_1,2)+np.power(s1_2,2)) + k
    s1 = sparse.hstack((sparse.diags(np.divide(s1_1,norm1),0),sparse.diags(np.divide(s1_2,norm1),0)),format = 'csr').T

    s2_1 = S11 - S22 - tmp
    s2_2 = 2*S21
    norm2 = np.sqrt(np.power(s2_1,2)+np.power(s2_2,2)) + k
    s2 = sparse.hstack((sparse.diags(np.divide(s2_1,norm2),0),sparse.diags(np.divide(s2_2,norm2),0)),format = 'csr').T

    trace = S11 + S22

    return s1,s2,trace

def smoothnessNE(g,m,n,kappa,sigma):
    # Computes the anisotropic image driven smoothness term
    # of Nagel and Enkelmann.
    # Parameters: g: image as m-by-n array
    #          kappa: regularization parameter
    # Returns: V: smoothness array

    # [Dx, Dy] = HS.forwardDifferenceImage(g,m,n)
    # grad = np.power(Dx,2) + np.power(Dy,2) # The square of the gradient
    # denom = grad + 2*math.pow(kappa,2) # Denominator in the NE diffusion matrix
    # sx = np.divide(np.power(Dx,2),denom) # gx^2/denominator
    # sy = np.divide(np.power(Dy,2),denom) # gy^2/denominator
    # sxy = np.divide(np.multiply(Dx,Dy),denom) # gx gy/denominator
    #
    # P_diag = np.hstack([sx,sy]) # The main diagonal in the diffusion matrix
    # P1 = sparse.diags([P_diag+math.pow(kappa,2),-sxy,-sxy],[0,m*n,-m*n])
    [s1,s2,trace] = dirDeriv(g,m,n,sigma)
    A = sparse.diags(np.divide(np.power(kappa,2)*np.ones(m*n),trace+2*np.power(kappa,2)*np.ones(m*n)),0)
    A = sparse.kron(sparse.eye(2),A)
    B = sparse.diags(np.divide(trace + np.power(kappa,2)*np.ones(m*n),trace+2*np.power(kappa,2)*np.ones(m*n)),0)
    B = sparse.kron(sparse.eye(2),B)
    P1 = A.dot(s1.dot(s1.T)) + B.dot(s2.dot(s2.T))
    P = sparse.kron(sparse.eye(2),P1,format = 'csr') # Diffusion matrix

    L = HS.makeLmatrix(m,n)

    V = ((L.T).dot(P)).dot(L)
    return V
