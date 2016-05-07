import computeColor as cc
import HS_method as HS
import NE_method as NE
import flowdriven_convex as FDC
import imageflowdriven as IF
import comp_reg as CR
import subquadraticData as SQ
import DataTermMethods as dtm

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math

# BC = Brightness Constraint
# GC = Gradient Contraint

method = 'IFD'
diff_method = 'forward'
data_penalize = 'quadratic'
# Boolean: True gives a separate penalization of BC and GC
separate_pen = False
# Regularization parameter for subquadratic functions
eps = 0.001
# Boolean: True normalizes the Data term
normalize = True
# regularization parameter for normalizing data term
zeta = 10
# Parameter for GC, gamma = 0 gives only BC
gamma = 0.0
# Std deviation for Gaussian convolution of reg tensor
mu_regTensor = 1.0
# Global scale regularization
# alpha = 3500.0
# regu = math.sqrt(1/alpha)
regu = 1
sigma_image = 0.3

savefigure =  0 # To save figure displayfigure must be TRUE
# savefig_dir = '/home/shomec/e/espenjv/Semester Project/Figures/'
savefig_dir =  'C:/Users/Espen/opticalflow/Report/Thesis/Figures/'
figure_title = 'Constraint Driven, normalized'
# Filename: 'generic' generates a filename from parameters and method
figure_filename = 'CR_normalized.png'
displayfigure = 1
# image_dir = 'C:/Users/Espen/opticalflow/Datasets/HamburgTaxi/'
# image_dir = 'C:/Users/Espen/opticalflow/Datasets/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/'
# image_dir = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/'
image_dir = 'C:/Users/Espen/opticalflow/Datasets/other-data-gray/RubberWhale/'

g1 = misc.imread(image_dir+'frame10.png')
g2 = misc.imread(image_dir+'frame11.png')
[m,n] = g1.shape
g1 = g1[-m/2:,0:n/2]
g2 = g2[-m/2:,0:n/2]
m,n = m/2,n/2

# g1 = misc.imread(image_dir+'taxi-00.tif')
# g2 = misc.imread(image_dir+'taxi-01.tif')
# g3 = misc.imread(image_dir+'taxi-02.tif')

# g1 = misc.imread(image_dir+'0000000000.png')
# g2 = misc.imread(image_dir+'0000000001.png')
# g3 = misc.imread(image_dir+'0000000002.png')

# g1 = misc.imresize(g1,25)
# g2 = misc.imresize(g2,25)
[m,n] = g1.shape

g1 = np.array(g1, dtype=np.double)
g2 = np.array(g2, dtype=np.double)
# g3 = np.array(g3, dtype=np.double)



# g = np.array([g1, g2, g3])


# sigma_image_time = 0.5
# g = ndimage.filters.gaussian_filter(g,[sigma_image_time,sigma_image_space,sigma_image_space])
g1 = ndimage.filters.gaussian_filter(g1,sigma_image)
g2 = ndimage.filters.gaussian_filter(g2,sigma_image)
# g1 = g[0]
# g2 = g[1]


g1 = np.reshape(g1.T,[1,m*n])[0]
g2 = np.reshape(g2.T,[1,m*n])[0]
# g3 = np.reshape(g3.T,[1,m*n])[0]

# Time discretization: Forward difference, time step = tau
tau = 1
g = g1

if method is 'HS':
    # Using Horn and Schunck Smoothness term
	if data_penalize is 'quadratic':
		V = HS.smoothnessHS(m,n)
		M,b = dtm.makeQuadraticDataTerm(g1,g2,m,n,diff_method,zeta,gamma,normalize)
		G = M + math.pow(regu,-2)*V
		[G,b] = HS.neumann_boundary(G,b,m,n)
		# Flow vector
		w = spsolve(G,b)
	if data_penalize is 'subquadratic':
		w = SQ.subquadraticData(g1,g2,m,n,regu,diff_method,gamma,separate_pen,zeta,eps,normalize)
elif method is 'ID':
	# Using Nagel and Enkelmann image driven method
	# Regularization parameter
	kappa = 1
	# D-matrix from model term
	# D = HS.makeDmatrix(g,m,n,diff_method)
	# Model term
	# M = (D.T).dot(D);
	[M,b] = dtm.makeQuadraticDataTerm(g1,g2,m,n,diff_method,zeta,gamma,normalize)
	# Smoothness term
	V = NE.smoothnessNE(g,m,n,kappa,mu_regTensor)
	# Model + smoothness
	G = M + math.pow(regu,-2)*V
	# RHS
	# b = -(D.T).dot(c)
	# Boundary conditions
	[G,b] = HS.neumann_boundary(G,b,m,n)
	# Flow vector
	w = spsolve(G,b)
elif method is 'FD':
    # Flow driven with convex penaliser
    w = FDC.findFlow(g,m,n,c,regu,diff_method)
elif method is 'IFD':
    # Image flow driven with convex penaliser
    w = IF.findFlow(g1,g2,m,n,regu,diff_method,data_penalize,separate_pen,mu_regTensor,zeta,eps,gamma,normalize)
elif method is 'CR':
	eta = 1
	w = CR.findFlow(g1,g2,m,n,regu,diff_method,mu_regTensor,data_penalize,zeta,eps,gamma,normalize)


u = w[0:m*n]
v = w[m*n:2*m*n]

if displayfigure:

	flow_image = cc.computeColor(np.reshape(u,[n,m]).T,np.reshape(v,[n,m]).T,m,n)/255
	plt.figure()
	plt.imshow(flow_image)
	plt.axis('off')
	if figure_title is 'generic':
		figure_title = 'Method: ' + method + 'Normalize:' + str(normalize) + ', regu: ' + str(regu)
	plt.title(figure_title)
	if savefigure:
		if figure_filename is 'generic':
			figure_filename = method + str(regu)[2:] + '.pdf'
		plt.savefig(savefig_dir+ figure_filename,bbox_inches = 'tight')
	plt.show()
