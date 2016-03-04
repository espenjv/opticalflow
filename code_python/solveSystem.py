import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

# g1 = cv2.imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-00.tif')
# g1 = cv2.imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-01.tif')
#
# Dx5 = cv2.Sobel(g1,-1,1,0,ksize=5)
# Dx3 = cv2.Sobel(g1,-1,1,0,ksize=3)
# Dy = cv2.Sobel(g1,-1,0,1,ksize=5)
#
#
#
# plt.figure()
# plt.imshow(Dx5, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#
# plt.figure()
# plt.imshow(Dx3, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

#
# cv2.imshow("sobel_image",Dx)
# cv2.waitKey(0)
# cv2.destroyAllWindows

regu = 0.003

g1 = misc.imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-00.tif')
g2 = misc.imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-01.tif')

g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)


ae = misc.imread('/home/shomec/e/espenjv/Pictures/ae.jpg')
print ae[:,:,1].shape
[m,n] = g1.shape

Gx = np.array([[-1 ,0, 1],[-2, 0 ,2],[-1, 0, 1]])
Gy = Gx.T


Dx = signal.convolve2d(g1,Gx,mode='same')
Dy = signal.convolve2d(g1,Gy,mode='same')

# Dx = ndimage.filters.sobel(g1,0,mode='constant')


Dx = np.reshape(Dx.T,[1,m*n])[0]
# print Dx[1:10]
Dx = sparse.diags(Dx,0,format='csr')
# Dy = ndimage.filters.sobel(g1,1,mode='constant')
Dy = np.reshape(Dy.T,[1,m*n])[0]
Dy = sparse.diags(Dy,0,format='csr')


D = sparse.hstack((Dx,Dy),format='csr')

c = np.subtract(np.reshape(g2.T,[1,m*n])[0],np.reshape(g1.T,[1,m*n])[0])


def smoothnessHS(m,n):
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m])
    # print len(np.nonzero(Lx)[0])
    # o = np.ones(m*n)
    # print np.sum(Lx.dot(o))
    Ly = sparse.kron(sparse.eye(n),sparse.diags([-np.ones(m), np.ones(m-1)],[0,1]),format = 'csr')

    L = sparse.kron(sparse.eye(2),sparse.vstack((Lx,Ly)),format = 'csr')

    l1 = L[0:m*n,0:m*n]
    l2 = L[m*n:3*m*n,0:2*m*n]
    l3 = L[3*m*n:4*m*n,0:2*m*n]

    print l3.shape
    o = np.ones(2*m*n)
    print np.sum(l3.dot(o))



    V = (L.T).dot(L)

    return V

M = (D.T).dot(D);

# plt.spy(M)
# plt.show()

V = smoothnessHS(m,n)

G = M + math.pow(regu,-2)*V

a = G.dot(np.ones(2*m*n))
#
# print G

d1 = G[0:m*n,0:m*n]
d2 = G[m*n:2*m*n,0:m*n]
d3 = G[0:m*n,m*n:2*m*n]
d4 = G[m*n:2*m*n,m*n:2*m*n]


# m1 = M[0:m*n,0:m*n]
# o = np.ones(m*n)
# print np.sum(m1.dot(o))

# v1 = V[0:m*n,0:m*n]
# o = np.ones(m*n)
# print np.sum(v1.dot(o))



# o = np.ones(m*n)
# print np.sum(d1.dot(o))
# print np.sum(d2.dot(o))
# print np.sum(d3.dot(o))
# print np.sum(d4.dot(o))
#
# # plt.spy(d1)
# # plt.show()
#
# print np.sum(a)

b = sparse.csr_matrix(-(D.T).dot(c))


w = spsolve(G,b)


u = w[0:m*n]
v = w[m*n:2*m*n]


flow = np.array([u,v])
temp = np.add(np.power(u,2),np.power(v,2))
f = np.sqrt(np.add(np.power(u,2),np.power(v,2)))

f = np.reshape(f,[n,m])

f = f.T


# u = np.reshape([w[0:m*n]],[m,n])
# v = np.reshape([w[m*n:2*m*n]],[m,n])


# print f.shape
#
# plt.imshow(f,'gray')
# plt.show()
#
#

#
# import cv2
#
# hsv = np.zeros((m,n,3))
# hsv[...,1] = 255
#
#
# mag, ang = cv2.cartToPolar(u, v)
# print ang
# hsv[...,0] = ang*180/np.pi/2
# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
# rgb = cv2.cvtColor(hsv,)
#
# cv2.imshow('frame2',rgb)
# cv2.waitKey()
#
#
# cv2.destroyAllWindows()
#
#


def makeColorwheel():
    RY = 15;
    YG = 6;
    GC = 4;
    CB = 11;
    BM = 13;
    MR = 6;

    ncols = RY + YG + GC + CB + BM + MR;

    colorwheel = np.zeros([ncols, 3]); # r g b

    col = 0;
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(RY)/RY).T
    col = col+RY;

    #YG
    colorwheel[col+np.arange(YG), 0] = 255 - np.floor(255*np.arange(YG)/YG).T;
    colorwheel[col+np.arange(YG), 1] = 255;
    col = col+YG;

    #GC
    colorwheel[col+np.arange(GC), 1] = 255;
    colorwheel[col+np.arange(GC), 2] = np.floor(255*np.arange(GC)/GC).T;
    col = col+GC;


    #CB
    colorwheel[col+np.arange(CB), 1] = 255 - np.floor(255*np.arange(CB)/CB).T;
    colorwheel[col+np.arange(CB), 2] = 255;
    col = col+CB;

    #BM
    colorwheel[col+np.arange(BM), 2] = 255;
    colorwheel[col+np.arange(BM), 0] = np.floor(255*np.arange(BM)/BM).T;
    col = col+BM;

    #MR
    colorwheel[col+np.arange(MR), 2] = 255 - np.floor(255*np.arange(MR)/MR).T;
    colorwheel[col+np.arange(MR), 0] = 255;

    return colorwheel


def computeColor(u,v):

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.add(np.power(u,2),np.power(v,2)))
    a = np.arctan2(-v, -u)/math.pi;
    fk = (a+1) /2 * (ncols-1) + 1;  #-1~1 maped to 1~ncols
    k0 = np.floor(fk);                 # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1==ncols+1] = 0;
    f = (fk - k0);
    k0 = np.array(k0,dtype=np.int)
    k1 = np.array(k1,dtype=np.int)

    i = 0
    img = np.zeros([m,n,3])
    print img.shape
    for i in np.arange(colorwheel.shape[1]):
        tmp = colorwheel[:,i];
        col0 = tmp[k0-1]/255;
        col1 = tmp[k1-1]/255;
        col = np.multiply(1-f,col0) + np.multiply(f,col1);

        idx = rad <= 1;
        col[idx] = 1-np.multiply(rad[idx],(1-col[idx]));    # increase saturation with radius

        col[~idx] = col[~idx]*0.75;             # out of range


        img[:,:, i] = np.array(np.floor(255*np.multiply(col,(1-nanIdx))),dtype=np.uint8);


    return img

flow_image = computeColor(np.reshape(u,[m,n]),np.reshape(v,[m,n]))

plt.imshow(flow_image)
plt.show()
