from scipy import sparse, signal
import numpy as np
from matplotlib import pyplot as plt
import math


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




def imageDiff(g,m,n):
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

def makeModelTerm(g1,g2,m,n,diff_method):
    k = 0.1
    g = g1

    if diff_method is 'forward':
        [gx,gy] = forwardDifferenceImage(g,m,n)
        [gxx,gxy] = forwardDifferenceImage(gx,m,n)
        [gyx,gyy] = forwardDifferenceImage(gy,m,n)

        # [g2x,g2y] = forwardDifferenceImage(g2,m,n)
        # [g2xx,g2xy] = forwardDifferenceImage(g2x,m,n)
        # [g2yx,g2yy] = forwardDifferenceImage(g2y,m,n)
    elif diff_method is 'central1':
        [gx, gy] = centralDifferenceImage1(g,m,n)
        [gxx, gxy] = centralDifferenceImage1(gx,m,n)
        [gyx, gyy] = centralDifferenceImage1(gy,m,n)
    elif diff_method is 'central2':
        [gx, gy] = centralDifferenceImage2(g,m,n)
        [gxx, gxy] = centralDifferenceImage2(gx,m,n)
        [gyx, gyy] = centralDifferenceImage2(gy,m,n)
    elif diff_method is 'sobel':
        [gx, gy] = imageDiff(g,m,n)
        [gxx, gxy] = imageDiff(gx,m,n)
        [gyx, gyy] = imageDiff(gy,m,n)

    gt = np.subtract(g2,g1)
    # gxt = np.subtract(g2x,gx)
    # gyt = np.subtract(g2y,gy)

    # Normalisation terms
    theta_0 = np.sqrt(np.power(gx,2) + np.power(gy,2) + k**2)
    theta_x = np.sqrt(np.power(gxx,2) + np.power(gxy,2) + k**2)
    theta_y = np.sqrt(np.power(gyx,2) + np.power(gyy,2) + k**2)

    gx = sparse.diags(np.divide(gx,theta_0),0,format='csr')
    gy = sparse.diags(np.divide(gy,theta_0),0,format='csr')
    gxx = sparse.diags(np.divide(gxx,theta_x),0,format='csr')
    gyx = sparse.diags(np.divide(gyx,theta_x),0,format='csr')
    gxy = sparse.diags(np.divide(gxy,theta_y),0,format='csr')
    gyy = sparse.diags(np.divide(gyy,theta_y),0,format='csr')

    grad_g = sparse.hstack((gx,gy),format='csr')
    grad_gx = sparse.hstack((gxx,gyx),format='csr')
    grad_gy = sparse.hstack((gxy,gyy),format='csr')


    #
    # # Model term
    M = (grad_g.T).dot(grad_g)
    #  + (grad_gx.T).dot(grad_gx) + (grad_gy.T).dot(grad_gy)
    # # RHS
    b = -M.dot((grad_g.T).dot(np.divide(gt,theta_0)))
    # -(grad_gx.T).dot(np.divide(gxt,theta_x)) -(grad_gy.T).dot(np.divide(gyt,theta_y))

    return M,b






def makeDmatrix(g,m,n,diff_method):
    # Forms the D-matrix in the model term discretization
    # Parameters: g: an image as array
    # Returns: D: 2-dimensional array of shape m*n x 2*m*n, with Dx and Dy as
    #           block diagonal matrices [Dx | Dy]

    if diff_method is 'forward':
        [dx,dy] = forwardDifferenceImage(g,m,n)
    elif diff_method is 'central1':
        [dx, dy] = centralDifferenceImage1(g,m,n)
    elif diff_method is 'central2':
        [dx, dy] = centralDifferenceImage2(g,m,n)
    elif diff_method is 'sobel':
        [dx, dy] = imageDiff(g,m,n)

    # [m,n] = g.shape
    # Dx = np.reshape(dx.T,[n,m]).T
    # plt.figure()
    # plt.imshow(Dx)
    # plt.show()
    #
    # Dy = np.reshape(dy.T,[n,m]).T
    # plt.figure()
    # plt.imshow(Dy)
    # plt.show()

    Dx = sparse.diags(dx,0,format='csr')
    Dy = sparse.diags(dy,0,format='csr')
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
    k = 0.00001
    # Lx = sparse.kron(sparse.eye(n),sparse.hstack((sparse.eye(m),np.zeros((m,m)))),format = 'csr') + sparse.kron(sparse.eye(n),sparse.hstack((np.zeros((m,m)),-sparse.eye(m))), format = 'csr')
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    Lx[:m,:2*m] = np.zeros(m,2*m)
    Lx[m*(n-1):m*n,m*(n-1):m*n] = np.zeros((m,m))
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    Ly[0,:2] = [0,0]
    Ly1[m-1,:] = np.zeros(m)
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')
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

def forwardDifferenceImage(g,m,n):
    #forwardDifferenceImage Computes approximation of the image gradient using
    #forward difference
    # Boundaries: zero first derivatives

    # TODO NEUMANN BOUNDARY CONDITIONS

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
