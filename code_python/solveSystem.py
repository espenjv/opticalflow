import computeColor as cc
import HS_method as HS
import NE_method as NE
import flowdriven_convex as FDC
import imageflowdriven as IF
import comp_reg as CR
import subquadraticData as SQ

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


method = 'HS'
diff_method = 'forward'
data_penalize = 'subquadratic'
savefigure =  1 # To save figure displayfigure must be TRUE
savefig_dir = '/home/shomec/e/espenjv/Semester Project/Figures/'
displayfigure = 1

# Global scale regularization
regu = 0.03
#image_dir = 'C:/Users/Espen/opticalflow/HamburgTaxi/'
image_dir = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/'

g1 = misc.imread(image_dir+'taxi-00.tif')
g2 = misc.imread(image_dir+'taxi-01.tif')
g3 = misc.imread(image_dir+'taxi-02.tif')


g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)
g3 = np.array(g3, dtype=np.double)

sigma = 0.5
g1 = ndimage.filters.gaussian_filter(g1,sigma)
g2 = ndimage.filters.gaussian_filter(g2,sigma)

[m,n] = g1.shape

g1 = np.reshape(g1.T,[1,m*n])[0]
g2 = np.reshape(g2.T,[1,m*n])[0]
g3 = np.reshape(g3.T,[1,m*n])[0]

# Time discretization: Forward difference, time step = tau
tau = 1
c = np.subtract(g2,g1)/(tau)
g = g1

if method is 'HS':
    # Using Horn and Schunck method
	# D-matrix from model term
	D = HS.makeDmatrix(g,m,n,diff_method)
	# Model term
	M = (D.T).dot(D);
    # Smoothness term
	V = HS.smoothnessHS(m,n)
    # Model + smoothness
	# M,b = HS.makeModelTerm(g1,g2,m,n,diff_method)
	G = M + math.pow(regu,-2)*V
    # RHS
	b = -(D.T).dot(c)
    # Flow vector
	w = spsolve(G,b)
elif method is 'subquad':
	w = SQ.subquadraticData(g,m,n,c,regu,diff_method)
elif method is 'ID':
	# Using Nagel and Enkelmann image driven method
	# Regularization parameter
	kappa = 1
	# D-matrix from model term
	D = HS.makeDmatrix(g,m,n,diff_method)
	# Model term
	M = (D.T).dot(D);
	# Smoothness term
	V = NE.smoothnessNE(g,m,n,kappa)
	# Model + smoothness
	G = M + math.pow(regu,-2)*V
	# RHS
	b = -(D.T).dot(c)
	# Flow vector
	w = spsolve(G,b)
elif method is 'FD':
    # Flow driven with convex penaliser
    w = FDC.findFlow(g,m,n,c,regu,diff_method)
elif method is 'IFD':
    # Image flow driven with convex penaliser
    w = IF.findFlow(g,m,n,c,regu,diff_method,data_penalize)
elif method is 'CR':
	eta = 1
	w = CR.findFlow(g,m,n,c,regu,diff_method)


u = w[0:m*n]
v = w[m*n:2*m*n]

if displayfigure:

	flow_image = cc.computeColor(np.reshape(u,[n,m]).T,np.reshape(v,[n,m]).T,m,n)/255
	plt.figure()
	plt.imshow(flow_image)
	plt.axis('off')
	plt.title('Method: ' + method + ', diff_method: ' + diff_method + ', regu: ' + str(regu))
	if savefigure:
		plt.savefig(savefig_dir+method + diff_method + str(regu) + '.png',bbox_inches = 'tight')
	plt.show()
