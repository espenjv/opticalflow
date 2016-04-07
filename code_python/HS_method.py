from scipy import sparse, signal
import numpy as np
from matplotlib import pyplot as plt
import math


def imageDiff(g):
    # Computes the Dx and Dy submatrices of the model term discretization.
    # Uses the sobel derivatives as approximation for the image gradients.
    # Parameters: g: an image as array
    # Returns: Dx: m*n 1-dimensional array with the approximation of the
    #               derivatives in the x-direction
    #          Dy: m*n 1-dimensional array with the approximation of the
    #               derivatives in the y-direction

    [m,n] = g.shape

    Gx = np.array([[-1 ,0, 1],[-2, 0 ,2],[-1, 0, 1]])
    Gy = Gx.T

    Dx = signal.convolve2d(g,Gx,mode='same')
    Dy = signal.convolve2d(g,Gy,mode='same')

    Dx = np.reshape(Dx.T,[1,m*n])[0]
    Dy = np.reshape(Dy.T,[1,m*n])[0]
    return Dx,Dy


def makeDmatrix(g):
    # Forms the D-matrix in the model term discretization
    # Parameters: g: an image as array
    # Returns: D: 2-dimensional array of shape m*n x 2*m*n, with Dx and Dy as
    #           block diagonal matrices [Dx | Dy]

    [Dx, Dy] = forwardDifferenceImage(g)
    #[Dx,Dy] = forwardDifferenceImage(g)
    Dx = sparse.diags(Dx,0,format='csr')
    Dy = sparse.diags(Dy,0,format='csr')
    D = sparse.hstack((Dx,Dy),format='csr')
    return D

def makeLmatrix(m,n):
    # Forms the L matrix used in the flow derivative approximation.
    # A matrix multipication with Lx and Ly gives the approximation of the
    # derivatives in x- and y-direction assuming neumann boundary conditions.
    # Parameters: m: number of rows in the image
    #             n: number of columns in the image
    # Returns: L: 2-dimensional array of shape 4mn x 2mn. The matrix
    #           multiplication grad_w= L[u,v].T gives a
    #           4mn vector grad_w = [u_x,u_y,v_x,v_y].T

    # Lx = sparse.kron(sparse.eye(n),sparse.hstack((sparse.eye(m),np.zeros((m,m)))),format = 'csr') + sparse.kron(sparse.eye(n),sparse.hstack((np.zeros((m,m)),-sparse.eye(m))), format = 'csr')
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    Lx[m*(n-1):m*n,m*(n-1):m*n] = np.zeros((m,m))
    print Lx.shape
    # TODO: Neumann boundary conditions on y derivative
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    Ly1[m-1,:] = np.zeros(m)
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')
    print Ly.shape
    L = sparse.kron(sparse.eye(2),sparse.vstack((Lx,Ly)),format = 'csr')
    return L

def smoothnessHS(m,n):
    # Computes the smoothness term of Horn and Schunck.
    # Parameters: m: number of rows in the image
    #             n: number of columns in the image
    # Returns: V: 2mn x 2mn array

    L = makeLmatrix(m,n)
    V = (L.T).dot(L)
    return V

def forwardDifferenceImage(g):
    #forwardDifferenceImage Computes approximation of the image gradient using
    #forward difference

    k = 0.01

    [m,n] = g.shape
    # TODO: Handle boundaries: fx and fy is the same for the two last columns and rows
    g_vec = np.reshape(g.T,[1,m*n])[0]
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    Lx[m*(n-1):m*n,m*(n-1):m*n] = sparse.eye(m)
    Lx[m*(n-1):m*n,m*(n-2):m*(n-1)] = -sparse.eye(m)
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    Ly1[m-1,:] = np.hstack((np.zeros(m-2),[-1,1]))
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')


    Dx = Lx.dot(g_vec) + math.pow(k,2)
    Dx
    Dy = Ly.dot(g_vec) + math.pow(k,2)

    return Dx,Dy

def centralDifferenceImage(g):
    [m,n] = g.shape
    Lx = sparse.diags([np.ones((n-2)*m),-8*np.ones((n-1)*m),8*np.ones((n-1)*m),-np.ones((n-2)*m)],[-2*m,-m,m,2*m])
    Ly = sparse.kron(sparse.eye(n),sparse.diags([np.ones((m-2)),-8*np.ones((m-1)),8*np.ones((m-1)),-np.ones((m-2))],[-2,-1,1,2]),format = 'csr')
    g_vec = np.reshape(g.T,[1,m*n])[0]

    Dx = Lx.dot(g_vec)/12
    Dy = Ly.dot(g_vec)/12

    return Dx,Dy
