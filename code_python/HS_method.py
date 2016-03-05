from scipy import sparse, signal
import numpy as np


def imageDiff(g):
    [m,n] = g.shape

    Gx = np.array([[-1 ,0, 1],[-2, 0 ,2],[-1, 0, 1]])
    Gy = Gx.T

    Dx = signal.convolve2d(g,Gx,mode='same')
    Dy = signal.convolve2d(g,Gy,mode='same')

    Dx = np.reshape(Dx.T,[1,m*n])[0]
    Dy = np.reshape(Dy.T,[1,m*n])[0]
    return Dx,Dy

def makeDmatrix(g):
    [Dx, Dy] = imageDiff(g)
    Dx = sparse.diags(Dx,0,format='csr')
    Dy = sparse.diags(Dy,0,format='csr')
    D = sparse.hstack((Dx,Dy),format='csr')
    return D

def makeLmatrix(m,n):
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m])
    Ly = sparse.kron(sparse.eye(n),sparse.diags([-np.ones(m), np.ones(m-1)],[0,1]),format = 'csr')
    L = sparse.kron(sparse.eye(2),sparse.vstack((Lx,Ly)),format = 'csr')
    return L

def smoothnessHS(m,n):
    L = makeLmatrix(m,n)
    V = (L.T).dot(L)
    return V
