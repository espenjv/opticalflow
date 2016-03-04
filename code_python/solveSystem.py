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

g1 = np.array(g1, dtype=np.int)
g2 = np.array(g2, dtype=np.int)


lena = misc.imread('/home/shomec/e/espenjv/Pictures/lena512.bmp')


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
    Lx = sparse.diags([np.ones(m*n),np.ones((n-1)*m)],[0,m])
    Ly = sparse.kron(sparse.eye(n),sparse.diags([-np.ones(m), np.ones(m-1)],[0,1]),format = 'csr')

    L = sparse.kron(sparse.eye(2),sparse.vstack((Lx,Ly)),format = 'csr')

    V = (L.T).dot(L)

    return V

M = (D.T).dot(D);

# plt.spy(M)
# plt.show()

V = smoothnessHS(m,n)

V2 = smoothnessHS(10,10)

G = M + math.pow(regu,-2)*V

b = sparse.csr_matrix(-(D.T).dot(c))


w = spsolve(G,b)


u = w[0:m*n]
v = w[m*n:2*m*n]

flow = np.array([u,v])

# u = np.reshape([w[0:m*n]],[m,n])
# v = np.reshape([w[m*n:2*m*n]],[m,n])

f = np.sqrt(np.power(u,2) + np.power(v,2))



A = np.array([[1, 3 ,4], [3, 5 ,9], [3, 4 ,1]])
A = sparse.csr_matrix(A)

x = np.array([2, 4 ,1])
x = sparse.csr_matrix(x)

#
# plt.imshow(f)
# plt.show()

import cv2

hsv = np.zeros((m,n,3))
hsv[...,1] = 255


mag, ang = cv2.cartToPolar(u, v)
print ang
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,)

cv2.imshow('frame2',rgb)
cv2.waitKey()


cv2.destroyAllWindows()


def computeColor(u,v):

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = sqrt(u.^2+v.^2)
    a = atan2(-v, -u)/pi;
    fk = (a+1) /2 * (ncols-1) + 1;  % -1~1 maped to 1~ncols
    k0 = floor(fk);                 % 1, 2, ..., ncols
    k1 = k0+1;
    k1(k1==ncols+1) = 1;
    f = fk - k0;


    for i = 1:size(colorwheel,2)
        tmp = colorwheel(:,i);
        col0 = tmp(k0)/255;
        col1 = tmp(k1)/255;
        col = (1-f).*col0 + f.*col1;

        idx = rad <= 1;
        col(idx) = 1-rad(idx).*(1-col(idx));    % increase saturation with radius

        col(~idx) = col(~idx)*0.75;             % out of range

        img(:,:, i) = uint8(floor(255*col.*(1-nanIdx)));
    end;

function colorwheel = makeColorwheel()
%   color encoding scheme
%   adapted from the color circle idea described at
%   http://members.shaw.ca/quadibloc/other/colint.htm
    RY = 15;
    YG = 6;
    GC = 4;
    CB = 11;
    BM = 13;
    MR = 6;

    ncols = RY + YG + GC + CB + BM + MR;

    colorwheel = zeros(ncols, 3); % r g b

    col = 0;
    %RY
    colorwheel(1:RY, 1) = 255;
    colorwheel(1:RY, 2) = floor(255*(0:RY-1)/RY)';
    col = col+RY;

    %YG
    colorwheel(col+(1:YG), 1) = 255 - floor(255*(0:YG-1)/YG)';
    colorwheel(col+(1:YG), 2) = 255;
    col = col+YG;

    %GC
    colorwheel(col+(1:GC), 2) = 255;
    colorwheel(col+(1:GC), 3) = floor(255*(0:GC-1)/GC)';
    col = col+GC;

    %CB
    colorwheel(col+(1:CB), 2) = 255 - floor(255*(0:CB-1)/CB)';
    colorwheel(col+(1:CB), 3) = 255;
    col = col+CB;

    %BM
    colorwheel(col+(1:BM), 3) = 255;
    colorwheel(col+(1:BM), 1) = floor(255*(0:BM-1)/BM)';
    col = col+BM;

    %MR
    colorwheel(col+(1:MR), 3) = 255 - floor(255*(0:MR-1)/MR)';
    colorwheel(col+(1:MR), 1) = 255;
