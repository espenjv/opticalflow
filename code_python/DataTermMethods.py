import imageDifferentiationMethods as idm
import numpy as np
from scipy import sparse

def MotionTerms(g1,g2,m,n,diff_method,zeta,normalize):
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
    if normalize:
        theta_0 = np.sqrt(np.power(gx,2) + np.power(gy,2) + zeta**2)
        theta_x = np.sqrt(np.power(gxx,2) + np.power(gxy,2) + zeta**2)
        theta_y = np.sqrt(np.power(gyx,2) + np.power(gyy,2) + zeta**2)
        gt = np.divide(gt,theta_0)
        gxt = np.divide(gxt,theta_x)
        gyt = np.divide(gyt,theta_y)

        gx = sparse.diags(np.divide(gx,theta_0),0,format='csr')
        gy = sparse.diags(np.divide(gy,theta_0),0,format='csr')
        gxx = sparse.diags(np.divide(gxx,theta_x),0,format='csr')
        gyx = sparse.diags(np.divide(gyx,theta_x),0,format='csr')
        gxy = sparse.diags(np.divide(gxy,theta_y),0,format='csr')
        gyy = sparse.diags(np.divide(gyy,theta_y),0,format='csr')
    else:
        gx = sparse.diags(gx,0,format='csr')
        gy = sparse.diags(gy,0,format='csr')
        gxx = sparse.diags(gxx,0,format='csr')
        gyx = sparse.diags(gyx,0,format='csr')
        gxy = sparse.diags(gxy,0,format='csr')
        gyy = sparse.diags(gyy,0,format='csr')



    grad_g = sparse.hstack((gx,gy),format='csr')
    grad_gx = sparse.hstack((gxx,gyx),format='csr')
    grad_gy = sparse.hstack((gxy,gyy),format='csr')

    return grad_g, grad_gx, grad_gy, gt, gxt, gyt

def makeQuadraticDataTerm(g1,g2,m,n,diff_method,zeta,gamma,normalize):

    grad_g, grad_gx, grad_gy, gt, gxt, gyt = MotionTerms(g1,g2,m,n,diff_method,zeta,normalize)
    #
    # # Model term
    M = (grad_g.T).dot(grad_g) + gamma*((grad_gx.T).dot(grad_gx) + (grad_gy.T).dot(grad_gy))
    # RHS
    b = - ((grad_g.T).dot(gt) + gamma*((grad_gx.T).dot(gxt) + (grad_gy.T).dot(gyt)))

    return M,b



def makeSubquadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize):
    g = g1

    grad_g, grad_gx, grad_gy, gt, gxt, gyt = MotionTerms(g1,g2,m,n,diff_method,zeta,normalize)

    J = (grad_g.T).dot(grad_g) + gamma*((grad_gx.T).dot(grad_gx) + (grad_gy.T).dot(grad_gy))

    grad3_g = sparse.hstack((grad_g,sparse.diags(gt,0)),format = 'csr')
    grad3_gx = sparse.hstack((grad_gx,sparse.diags(gxt,0)),format = 'csr')
    grad3_gy = sparse.hstack((grad_gy,sparse.diags(gyt,0)),format = 'csr')
    # print grad3_g.shape
    J3 = (grad3_g.T).dot(grad3_g) + gamma*((grad3_gx.T).dot(grad3_gx) + (grad3_gy.T).dot(grad3_gy))
    w3 = sparse.hstack((w,np.ones(m*n)))

    # print  (w3).dot(J3).dot(w3.T)[0]
    # Convex penaliser
    psi_deriv = sparse.diags(np.divide(np.ones(m*n),np.sqrt((w3).dot(J3).dot(w3.T)[0,0] +np.power(eps,2))),0)
    psi_d = sparse.kron(sparse.eye(2),psi_deriv)

    b = -psi_d.dot(((grad_g.T).dot(gt) + gamma*((grad_gx.T).dot(gxt) + (grad_gy.T).dot(gyt))))
    M = psi_d.dot(J)

    return M,b

def makeSeparateSubquadraticDataTerm(w,g1,g2,m,n,diff_method,zeta,eps,gamma,normalize):
    g = g1

    grad_g, grad_gx, grad_gy, gt, gxt, gyt = MotionTerms(g1,g2,m,n,diff_method,zeta,normalize)

    J0 = (grad_g.T).dot(grad_g)
    Jxy = gamma*((grad_gx.T).dot(grad_gx) + (grad_gy.T).dot(grad_gy))

    grad3_g = sparse.hstack((grad_g,sparse.diags(gt,0)),format = 'csr')
    grad3_gx = sparse.hstack((grad_gx,sparse.diags(gxt,0)),format = 'csr')
    grad3_gy = sparse.hstack((grad_gy,sparse.diags(gyt,0)),format = 'csr')
    # print grad3_g.shape
    J30 = (grad3_g.T).dot(grad3_g)
    J3xy = gamma*((grad3_gx.T).dot(grad3_gx) + (grad3_gy.T).dot(grad3_gy))
    w3 = sparse.hstack((w,np.ones(m*n)))

    # print  (w3).dot(J3).dot(w3.T)[0]
    # Convex penaliser
    psi_deriv0 = sparse.diags(np.divide(np.ones(m*n),np.sqrt((w3).dot(J30).dot(w3.T)[0,0] +np.power(eps,2))),0)
    psi_d0 = sparse.kron(sparse.eye(2),psi_deriv0)

    psi_derivxy = sparse.diags(np.divide(np.ones(m*n),np.sqrt((w3).dot(J3xy).dot(w3.T)[0,0] +np.power(eps,2))),0)
    psi_dxy = sparse.kron(sparse.eye(2),psi_derivxy)

    b = -psi_d0.dot((grad_g.T).dot(gt)) + gamma*psi_dxy.dot((grad_gx.T).dot(gxt) + (grad_gy.T).dot(gyt))
    M = psi_d0.dot(J0) + psi_dxy.dot(Jxy)

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
