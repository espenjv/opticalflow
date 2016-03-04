import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage, signal
import math



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
print Dx
Dx = np.diag(Dx,0)
# Dy = ndimage.filters.sobel(g1,1,mode='constant')
Dy = np.reshape(Dy.T,[1,m*n])[0]
print Dy
Dy = np.diag(Dy,0)


D = np.hstack((Dx,Dy))

c = np.subtract(np.reshape(g2.T,[1,m*n])[0],np.reshape(g1.T,[1,m*n])[0])


def smoothnessHS(m,n):
    Lx = np.diag(np.ones(m*n),0) + np.diag(np.ones((n-1)*m),m)
    Ly = np.kron(np.eye(n), np.diag(-np.ones(m),0) + np.diag(np.ones(m-1),1))

    L = np.kron(np.eye(2),np.vstack((Lx,Ly)))

    V = (L.T).dot(L)

    return V

M = (D.T).dot(D);

# plt.spy(M)
# plt.show()

V = smoothnessHS(m,n)

V2 = smoothnessHS(10,10)

G = M + math.pow(regu,-2)*V

b = -(D.T).dot(c)


w = spsolve(G,b)


u = w[0:m*n]
v = w[m*n:2*m*n]


u = np.reshape([w[0:m*n]],[m,n])
v = np.reshape([w[m*n:2*m*n]],[m,n])

f = np.sqrt(np.power(u,2) + np.power(v,2))

print w

plt.spy(G)
plt.show()

#
# plt.imshow(f)
# plt.show()
