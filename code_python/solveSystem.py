import computeColor as cc
import HS_method as HS
import NE_method as NE
import flowdriven_convex as FDC
import imageflowdriven as IF

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


method = 'ID'
savefigure = 0 # To save figure displayfigure must be TRUE
displayfigure = 1

regu = 0.003

image_dir = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/'

g1 = misc.imread(image_dir+'taxi-00.tif')
g2 = misc.imread(image_dir+'taxi-01.tif')

g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)


[m,n] = g1.shape
c = np.subtract(np.reshape(g2.T,[1,m*n])[0],np.reshape(g1.T,[1,m*n])[0])
D = HS.makeDmatrix(g1)
M = (D.T).dot(D);
if method is 'HS':
    V = HS.smoothnessHS(m,n)
    G = M + math.pow(regu,-2)*V
    b = sparse.csr_matrix(-(D.T).dot(c))
    w = spsolve(G,b)
elif method is 'ID':
    kappa = 1
    V = NE.smoothnessNE(g1,kappa)
    G = M + math.pow(regu,-2)*V
    b = sparse.csr_matrix(-(D.T).dot(c))
    w = spsolve(G,b)
elif method is 'FD':
    w = FDC.findFlow(g1,c,regu)
elif method is 'IFD':
    w = IF.findFlow(g1,c,regu)


u = w[0:m*n]
v = w[m*n:2*m*n]

if displayfigure:

    flow_image = cc.computeColor(np.reshape(u,[n,m]).T,np.reshape(v,[n,m]).T,m,n)/255

    plt.figure()
    plt.imshow(flow_image)
    plt.title('Method: ' + method)
    plt.show()
    if savefigure:
        plt.savefig(method +'.png',bbox_inches = 'tight')
