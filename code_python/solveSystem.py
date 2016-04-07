import computeColor as cc
import HS_method as HS
import NE_method as NE
import flowdriven_convex as FDC
import imageflowdriven as IF
import comp_reg as CR

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


method = 'CR'
savefigure = 0 # To save figure displayfigure must be TRUE
displayfigure = 1

# Global scale regularization
regu = 0.03
#image_dir = 'C:/Users/Espen/opticalflow/HamburgTaxi/'
image_dir = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/'

g1 = misc.imread(image_dir+'taxi-00.tif')
g2 = misc.imread(image_dir+'taxi-01.tif')

g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)


[m,n] = g1.shape
# Time discretization: Forward difference, time step = 1
c = np.subtract(np.reshape(g2.T,[1,m*n])[0],np.reshape(g1.T,[1,m*n])[0])
# D-matrix from model term
D = HS.makeDmatrix(g1)
# Model term
M = (D.T).dot(D);
if method is 'HS':
    # Using Horn and Schunck method
    # Smoothness term
	V = HS.smoothnessHS(m,n)
    # Model + smoothness
	G = M + math.pow(regu,-2)*V
    # RHS
	b = -(D.T).dot(c)
    # Flow vector
	w = spsolve(G,b)
elif method is 'ID':
    # Using Nagel and Enkelmann image driven method
    # Regularization parameter
    kappa = 1
    # Smoothness term
    V = NE.smoothnessNE(g1,kappa)
    # Model + smoothness
    G = M + math.pow(regu,-2)*V
    # RHS
    b = -(D.T).dot(c)
    # Flow vector
    w = spsolve(G,b)
elif method is 'FD':
    # Flow driven with convex penaliser
    w = FDC.findFlow(g1,c,regu)
elif method is 'IFD':
    # Image flow driven with convex penaliser
    w = IF.findFlow(g1,c,regu)
elif method is 'CR':
	eta = 0.7
	sigma = math.sqrt(2)/(4*eta)
	g = eta*g1
	g = ndimage.filters.gaussian_filter(g,sigma)
	w = CR.findFlow(g,c,regu)


u = w[0:m*n]
v = w[m*n:2*m*n]

if displayfigure:

	flow_image = cc.computeColor(np.reshape(u,[n,m]).T,np.reshape(v,[n,m]).T,m,n)/255
	plt.figure()
	plt.imshow(flow_image)
	plt.axis('off')
	plt.title('Method: ' + method +' regu: ' + str(regu))
	if savefigure:
		plt.savefig(method + str(regu) + '.png',bbox_inches = 'tight')
	plt.show()
