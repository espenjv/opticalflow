from scipy import sparse, signal
import numpy as np
import math



def sobelDerivative(g,m,n):
    # Computes the Dx and Dy submatrices of the model term discretization.
    # Uses the sobel derivatives as approximation for the image gradients.
    # Parameters: g: an image as array
    # Returns: Dx: m*n 1-dimensional array with the approximation of the
    #               derivatives in the x-direction
    #          Dy: m*n 1-dimensional array with the approximation of the
    #               derivatives in the y-direction

    g = np.reshape(g,[n,m]).T

    Gx = np.array([[-1 ,0, 1],[-2, 0 ,2],[-1, 0, 1]])
    Gy = Gx.T

    Dx = signal.convolve2d(g,Gx,mode='same')
    Dy = signal.convolve2d(g,Gy,mode='same')

    Dx = np.reshape(Dx.T,[1,m*n])[0]
    Dy = np.reshape(Dy.T,[1,m*n])[0]
    return Dx,Dy


def forwardDifferenceImage(g,m,n):
    #forwardDifferenceImage Computes approximation of the image gradient using
    #forward difference
    # Boundaries: zero first derivatives


    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    Lx[m*(n-1):m*n,m*(n-1):m*n] = np.zeros((m,m))
    Lx[m*(n-1):m*n,m*(n-2):m*(n-1)] = np.zeros((m,m))
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    Ly1[m-1,:] = np.hstack((np.zeros(m-2),[0,0]))
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')


    Dx = Lx.dot(g)
    Dy = Ly.dot(g)

    return Dx,Dy

def backwardDifferenceImage(g,m,n):

    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    Lx[m*(n-1):m*n,m*(n-2):m*(n-1)] = np.zeros((m,m))
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    Ly1[0,:] = np.hstack((np.zeros(m-2),[0,0]))
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')


    Dx = (-Lx.T).dot(g)
    Dy = (-Ly.T).dot(g)

    return Dx,Dy

def centralDifferenceImage1(g):
    #forwardDifferenceImage Computes approximation of the image gradient using
    #forward difference


    lx = sparse.hstack((-sparse.eye(m),np.zeros((m,m))))
    lx = sparse.hstack((lx,sparse.eye(m)))
    lx = 1.0/2*lx
    Lx = 1.0/2*sparse.diags([-np.ones(m*(n-1)),np.ones((n-1)*m)],[-m,m],format = 'lil')
    Lx[m*(n-1):m*n,m*(n-3):m*n] = lx
    Lx[0:m,0:m*3] = lx

    ly = np.array([-1.0,0,1.0])/2
    Ly1 = sparse.diags([-np.ones(m-1), np.ones(m-1)],[-1,1],format = 'lil')
    Ly1[m-1:m,m-3:m] = ly
    Ly1[0,0:3] = ly
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')


    Dx = Lx.dot(g)
    Dx
    Dy = Ly.dot(g)

    return Dx,Dy

def centralDifferenceImage2(g):


    lx = sparse.hstack((sparse.eye(m),-8*sparse.eye(m)))
    lx = sparse.hstack((lx,np.zeros((m,m))))
    lx = sparse.hstack((lx,8*sparse.eye(m)))
    lx = sparse.hstack((lx,-sparse.eye(m)))
    Lx = sparse.diags([np.ones((n-2)*m),-8*np.ones((n-1)*m),8*np.ones((n-1)*m),-np.ones((n-2)*m)],[-2*m,-m,m,2*m],format='lil')
    Lx[m*(n-1):m*n,m*(n-5):m*n] = lx
    Lx[0:m,0:m*5] = lx
    Lx = sparse.csr_matrix(Lx)

    ly = np.array([-8.0,1.0,0,8.0,-1.0])
    Ly1 = sparse.diags([np.ones(m-2),-8*np.ones(m-1),8*np.ones(m-1),-np.ones(m-2)],[-2,-1,1,2],format = 'lil')
    Ly1[m-1,m-5:m] = ly
    Ly1[0,0:5] = ly
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')


    Dx = Lx.dot(g)/12
    Dy = Ly.dot(g)/12

    return Dx,Dy

def secondOrderDerivatives(g):

    k = 0.01

    [m,n] = g.shape
    g_vec = np.reshape(g.T,[1,m*n])[0]
    lxx = sparse.hstack((-sparse.eye(m),sparse.eye(m)))
    Lxx = sparse.diags([np.ones(m*(n-1)),-2*np.ones(n*m),np.ones((n-1)*m)],[-m,0,m],format = 'lil')
    Lxx[m*(n-1):m*n,m*(n-2):m*n] = -lxx
    Lxx[0:m,0:m*2] = lxx

    lyy = np.array([-1.0,1.0])
    Lyy1 = sparse.diags([np.ones(m-1), -2*np.ones(m),np.ones(m-1)],[-1,0,1],format = 'lil')
    Lyy1[m-1:m,m-2:m] = -lyy
    Lyy1[0,0:2] = lyy
    Lyy = sparse.kron(sparse.eye(n),Lyy1,format = 'csr')

    Lxy = sparse.diags(-2*np.ones(m*n),0,format = 'lil')
    Lxy[0,0] = -1
    Lxy[m*n-1,m*n-1] = -1

    sub_1= sparse.diags(np.ones(m-1),1,format = 'lil')
    sub_1[m-1,m-1] = 1
    sub1 = sparse.kron(sparse.diags(np.ones(n-1),1),sub_1)

    sub_2 = sparse.diags(np.ones(m-1),-1,format = 'lil')
    sub_2[0,0] = 1
    sub2 = sparse.kron(sparse.diags(np.ones(n-1),-1),sub_2)

    sub3 =  sparse.diags(np.ones(m-1),-1,format = 'lil')
    sub4 = sparse.diags(np.ones(m-1),1,format = 'lil')

    Lxy[0:m,0:m] = Lxy[0:m,0:m] + sub3
    Lxy[m*(n-1):m*n,m*(n-1):m*n] = Lxy[m*(n-1):m*n,m*(n-1):m*n] + sub4

    Lxy = Lxy + sub1 + sub2


    Dxx = Lxx.dot(g_vec) + math.pow(k,2)
    Dyy = Lyy.dot(g_vec) + math.pow(k,2)

    Dxy = 1/2*(Lxy.dot(g_vec) + Dxx + Dyy)

    return Dxx,Dyy,Dxy
