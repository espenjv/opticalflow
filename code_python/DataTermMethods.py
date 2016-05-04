import imageDifferentiationMethods as idm
import numpy as np
from scipy import sparse

def MotionTerms(g1,g2,m,n,diff_method):
    k = 0.00001
    g = g1

    if diff_method is 'forward':
        [gx,gy] = idm.forwardDifferenceImage(g,m,n)
        [gxx,gxy] = idm.forwardDifferenceImage(gx,m,n)
        [gyx,gyy] = idm.forwardDifferenceImage(gy,m,n)

        [g2x,g2y] = idm.forwardDifferenceImage(g2,m,n)
        [g2xx,g2xy] = idm.forwardDifferenceImage(g2x,m,n)
        [g2yx,g2yy] = idm.forwardDifferenceImage(g2y,m,n)
    elif diff_method is 'central1':
        [gx, gy] = idm.centralDifferenceImage1(g,m,n)
        [gxx, gxy] = idm.centralDifferenceImage1(gx,m,n)
        [gyx, gyy] = idm.centralDifferenceImage1(gy,m,n)
    elif diff_method is 'central2':
        [gx, gy] = idm.centralDifferenceImage2(g,m,n)
        [gxx, gxy] = idm.centralDifferenceImage2(gx,m,n)
        [gyx, gyy] = idm.centralDifferenceImage2(gy,m,n)
    elif diff_method is 'sobel':
        [gx, gy] = idm.imageDiff(g,m,n)
        [gxx, gxy] = idm.imageDiff(gx,m,n)
        [gyx, gyy] = idm.imageDiff(gy,m,n)

    gt = np.subtract(g2,g1)
    gxt = np.subtract(g2x,gx)
    gyt = np.subtract(g2y,gy)

    # plt.plot(np.sqrt(np.power(gx,2) + np.power(gy,2)))

    # Normalisation terms
    theta_0 = np.sqrt(np.power(gx,2) + np.power(gy,2) + k**2)
    theta_x = np.sqrt(np.power(gxx,2) + np.power(gxy,2) + k**2)
    theta_y = np.sqrt(np.power(gyx,2) + np.power(gyy,2) + k**2)

    gt = np.divide(gt,theta_0)
    gxt = np.divide(gxt,theta_x)
    gyt = np.divide(gyt,theta_y)

    gx = sparse.diags(np.divide(gx,theta_0),0,format='csr')
    gy = sparse.diags(np.divide(gy,theta_0),0,format='csr')
    gxx = sparse.diags(np.divide(gxx,theta_x),0,format='csr')
    gyx = sparse.diags(np.divide(gyx,theta_x),0,format='csr')
    gxy = sparse.diags(np.divide(gxy,theta_y),0,format='csr')
    gyy = sparse.diags(np.divide(gyy,theta_y),0,format='csr')



    grad_g = sparse.hstack((gx,gy),format='csr')
    grad_gx = sparse.hstack((gxx,gyx),format='csr')
    grad_gy = sparse.hstack((gxy,gyy),format='csr')

    return grad_g, grad_gx, grad_gy, gt, gxt, gyt

def makeModelTerm(g1,g2,m,n,diff_method,gamma):
    k = 0.00001
    g = g1

    if diff_method is 'forward':
        [gx,gy] = idm.forwardDifferenceImage(g,m,n)
        [gxx,gxy] = idm.forwardDifferenceImage(gx,m,n)
        [gyx,gyy] = idm.forwardDifferenceImage(gy,m,n)

        [g2x,g2y] = idm.forwardDifferenceImage(g2,m,n)
        [g2xx,g2xy] = idm.forwardDifferenceImage(g2x,m,n)
        [g2yx,g2yy] = idm.forwardDifferenceImage(g2y,m,n)
    elif diff_method is 'central1':
        [gx, gy] = idm.centralDifferenceImage1(g,m,n)
        [gxx, gxy] = idm.centralDifferenceImage1(gx,m,n)
        [gyx, gyy] = idm.centralDifferenceImage1(gy,m,n)
    elif diff_method is 'central2':
        [gx, gy] = idm.centralDifferenceImage2(g,m,n)
        [gxx, gxy] = idm.centralDifferenceImage2(gx,m,n)
        [gyx, gyy] = idm.centralDifferenceImage2(gy,m,n)
    elif diff_method is 'sobel':
        [gx, gy] = idm.imageDiff(g,m,n)
        [gxx, gxy] = idm.imageDiff(gx,m,n)
        [gyx, gyy] = idm.imageDiff(gy,m,n)

    gt = np.subtract(g2,g1)
    gxt = np.subtract(g2x,gx)
    gyt = np.subtract(g2y,gy)

    # plt.plot(np.sqrt(np.power(gx,2) + np.power(gy,2)))

    # Normalisation terms
    theta_0 = np.sqrt(np.power(gx,2) + np.power(gy,2) + k**2)
    theta_x = np.sqrt(np.power(gxx,2) + np.power(gxy,2) + k**2)
    theta_y = np.sqrt(np.power(gyx,2) + np.power(gyy,2) + k**2)

    gt = np.divide(gt,theta_0)
    gxt = np.divide(gxt,theta_x)
    gyt = np.divide(gyt,theta_y)

    gx = sparse.diags(np.divide(gx,theta_0),0,format='csr')
    gy = sparse.diags(np.divide(gy,theta_0),0,format='csr')
    gxx = sparse.diags(np.divide(gxx,theta_x),0,format='csr')
    gyx = sparse.diags(np.divide(gyx,theta_x),0,format='csr')
    gxy = sparse.diags(np.divide(gxy,theta_y),0,format='csr')
    gyy = sparse.diags(np.divide(gyy,theta_y),0,format='csr')

    J11 = np.reshape(np.divide(np.power(gx,2),theta_0) + np.divide(np.power(gxx,2),theta_x) + np.divide(np.power(Dyx,2),theta_y),[n,m]).T
    R11 = ndimage.filters.gaussian_filter(R11,sigma)
    R11 = np.reshape(R11.T,[1,m*n])[0]
    R12 = np.reshape(np.divide(np.multiply(Dx,Dy),theta_0) + np.divide(np.multiply(Dxx,Dxy),theta_x) + np.divide(np.multiply(Dyy,Dyx),theta_y),[n,m]).T
    R12 = ndimage.filters.gaussian_filter(R12,sigma)
    R12 = np.reshape(R12.T,[1,m*n])[0]
    R21 = R12
    R22 = np.reshape(np.divide(np.power(Dy,2),theta_0) + np.divide(np.power(Dxy,2),theta_x) + np.divide(np.power(Dyx,2),theta_y),[n,m]).T
    R22 = ndimage.filters.gaussian_filter(R22,sigma)
    R22 = np.reshape(R22.T,[1,m*n])[0]


    grad_g, grad_gx, grad_gy, gt, gxt, gyt = MotionTerms(g1,g2,m,n,diff_method)
    #
    # # Model term
    M = grad_g.dot(grad_g.T) + gamma*(grad_gx.dot(grad_gx.T) + grad_gy.dot(grad_gy.T)))
    # RHS
    b = - ((grad_g + grad_gx + grad_gy).T).dot(gt + gamma*(gxt + gyt))

    return M,b





def makeSubquadratic(w,m,n,c,D,eps):

    # Convex penaliser
    psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt(np.power(D.dot(w)+ c,2) +math.pow(eps,2))),0)
    psi_d = sparse.kron(sparse.eye(2),psi_deriv)
    b = -psi_d.dot((D.T).dot(c))

    M = psi_d.dot((D.T).dot(D))

    return M,b


def makeDmatrix(g,m,n,diff_method):
    # Forms the D-matrix in the model term discretization
    # Parameters: g: an image as array
    # Returns: D: 2-dimensional array of shape m*n x 2*m*n, with Dx and Dy as
    #           block diagonal matrices [Dx | Dy]

    if diff_method is 'forward':
        [dx,dy] = idm.forwardDifferenceImage(g,m,n)
    elif diff_method is 'central1':
        [dx, dy] = idm.centralDifferenceImage1(g,m,n)
    elif diff_method is 'central2':
        [dx, dy] = idm.centralDifferenceImage2(g,m,n)
    elif diff_method is 'sobel':
        [dx, dy] = idm.imageDiff(g,m,n)

    Dx = sparse.diags(dx,0,format='csr')
    Dy = sparse.diags(dy,0,format='csr')
    D = sparse.hstack((Dx,Dy),format='csr')
    return D
