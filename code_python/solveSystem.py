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
data_penalize = 'quadratic'
MotionTerm = 'GCA'

savefigure =  0 # To save figure displayfigure must be TRUE
savefig_dir = '/home/shomec/e/espenjv/Semester Project/Figures/'
displayfigure = 1

# Global scale regularization
regu = 0.1
#image_dir = 'C:/Users/Espen/opticalflow/HamburgTaxi/'
image_dir = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/'

gamma = 0.01
sigma_regTensor = 0

g1 = misc.imread(image_dir+'taxi-00.tif')
g2 = misc.imread(image_dir+'taxi-01.tif')
g3 = misc.imread(image_dir+'taxi-02.tif')


g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)
g3 = np.array(g3, dtype=np.double)

# g = np.array([g1, g2, g3])

sigma_image = 0.5
# sigma_image_time = 0.5
# g = ndimage.filters.gaussian_filter(g,[sigma_image_time,sigma_image_space,sigma_image_space])
g1 = ndimage.filters.gaussian_filter(g1,sigma_image)
g2 = ndimage.filters.gaussian_filter(g2,sigma_image)
# g1 = g[0]
# g2 = g[1]

[m,n] = g1.shape

g1 = np.reshape(g1.T,[1,m*n])[0]
g2 = np.reshape(g2.T,[1,m*n])[0]
g3 = np.reshape(g3.T,[1,m*n])[0]

# Time discretization: Forward difference, time step = tau
tau = 1
g = g1

if method is 'HS':
    # Using Horn and Schunck Smoothness term
	if data_penalize is 'quadratic':
		V = HS.smoothnessHS(m,n)
		if MotionTerm is 'BCA':
			# D-matrix from model term
			D = HS.makeDmatrix(g,m,n,diff_method)
			# Model term
			M = (D.T).dot(D);
			# Model + smoothness
			G = M + math.pow(regu,-2)*V
			# RHS
			gt = np.subtract(g2,g1)/(tau)
			b = -(D.T).dot(gt)
			[G,b] = HS.neumann_boundary(G,b,m,n)
			# Flow vector
			w = spsolve(G,b)
		elif MotionTerm is 'GCA':
			M,b = HS.makeModelTerm(g1,g2,m,n,diff_method,gamma)
			G = M + math.pow(regu,-2)*V
			[G,b] = HS.neumann_boundary(G,b,m,n)
			# Flow vector
			w = spsolve(G,b)
	if data_penalize is 'subquadratic':
		w = SQ.subquadraticData(g1,g2,m,n,regu,diff_method,MotionTerm,gamma)
elif method is 'ID':
	# Using Nagel and Enkelmann image driven method
	# Regularization parameter
	kappa = 1
	# D-matrix from model term
	D = HS.makeDmatrix(g,m,n,diff_method)
	# Model term
	M = (D.T).dot(D);
	# Smoothness term
	V = NE.smoothnessNE(g,m,n,kappa,sigma_regTensor)
	# Model + smoothness
	G = M + math.pow(regu,-2)*V
	# RHS
	b = -(D.T).dot(c)
	# Boundary conditions
	[G,b] = HS.neumann_boundary(G,b,m,n)
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
		plt.savefig(savefig_dir+method + diff_method + str(regu)[2:] + '.pdf',bbox_inches = 'tight')
	plt.show()
