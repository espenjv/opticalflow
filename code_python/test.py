import computeColor as cc
import HS_method as HS
import NE_method as NE
import flowdriven_convex as FDC
import imageflowdriven as IF
import multigrid as mg

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage,sparse, signal
from scipy.sparse.linalg import spsolve
import math


method = 'FD'
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

c = np.subtract(np.reshape(g2.T,[1,m*n])[0],np.reshape(g1.T,[1,m*n])[0])


g = g1
#
# Ly = sparse.kron(sparse.eye(n),sparse.diags([-np.ones(m), np.ones(m-1)],[0,1]),format = 'csr')
# R_sub1 = sparse.kron(sparse.eye(m/2),[1.0/16, 1.0/8]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/16,0])
# R_sub2 = sparse.kron(sparse.eye(m/2),[1.0/8, 1.0/4]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/8,0])
# R1 = sparse.hstack((R_sub1,R_sub2))
# R = sparse.kron(sparse.eye(n/2),R1)+ sparse.kron(sparse.diags(np.ones(n/2-1),1),sparse.hstack((R_sub1,np.zeros((m/2,m)))))
# R = sparse.kron(sparse.eye(2),R,format = 'csr')
# Ly_1 = sparse.kron(sparse.eye(m*n),[1, -1])
# # plt.spy(Ly_1)
# # plt.show()
# Lx = sparse.kron(sparse.eye(n),sparse.hstack((sparse.eye(m),np.zeros((m,m))))) + sparse.kron(sparse.eye(n),sparse.hstack((np.zeros((m,m)),-sparse.eye(m))))
#
# Ly = sparse.kron(sparse.eye(n),sparse.hstack((sparse.eye(m),np.zeros((m,m)))),format = 'csr') + sparse.kron(sparse.eye(n),sparse.hstack((np.zeros((m,m)),-sparse.eye(m))), format = 'csr')
# print Lx.shape
# plt.spy(Lx)
# plt.show()

# g_vec = np.reshape(g.T,[1,m*n])[0]
# lx = sparse.hstack((-sparse.eye(m),np.zeros((m,m))))
# lx = sparse.hstack((lx,sparse.eye(m)))
# lx = 1.0/2*lx
# Lx = 1.0/2*sparse.diags([-np.ones(m*(n-1)),np.ones((n-1)*m)],[-m,m],format = 'lil')
# Lx[m*(n-1):m*n,m*(n-3):m*n] = lx
# Lx[0:m,0:m*3] = lx
#
# dx = Lx.dot(g_vec)
# Dx = np.reshape(dx.T,[n,m]).T
# plt.figure()
# plt.imshow(Dx)
# plt.show()
# sub_1 = sparse.lil_matrix((m,2*m))
# k = 0.001
#
# g_vec = np.reshape(g.T,[1,m*n])[0]
# lxx = sparse.hstack((-sparse.eye(m),sparse.eye(m)))
# Lxx = sparse.diags([np.ones(m*(n-1)),-2*np.ones(n*m),np.ones((n-1)*m)],[-m,0,m],format = 'lil')
# Lxx[m*(n-1):m*n,m*(n-2):m*n] = -lxx
# Lxx[0:m,0:m*2] = lxx
#
# lyy = np.array([-1.0,1.0])
# Lyy1 = sparse.diags([np.ones(m-1), -2*np.ones(m),np.ones(m-1)],[-1,0,1],format = 'lil')
# Lyy1[m-1:m,m-2:m] = -lyy
# Lyy1[0,0:2] = lyy
# Lyy = sparse.kron(sparse.eye(n),Lyy1,format = 'csr')
#
# #TODO: fxy derivative. See formula in blue book
# Lxy = sparse.diags(-2*np.ones(m*n),0,format = 'lil')
# Lxy[0,0] = -1
# Lxy[m*n-1,m*n-1] = -1
#
# sub_1= sparse.diags(np.ones(m-1),1,format = 'lil')
# sub_1[m-1,m-1] = 1
# sub1 = sparse.kron(sparse.diags(np.ones(n-1),1),sub_1)
#
# sub_2 = sparse.diags(np.ones(m-1),-1,format = 'lil')
# sub_2[0,0] = 1
# sub2 = sparse.kron(sparse.diags(np.ones(n-1),-1),sub_2)
#
# sub3 =  sparse.diags(np.ones(m-1),-1,format = 'lil')
# sub4 = sparse.diags(np.ones(m-1),1,format = 'lil')
#
# Lxy[0:m,0:m] = Lxy[0:m,0:m] + sub3
# Lxy[m*(n-1):m*n,m*(n-1):m*n] = Lxy[m*(n-1):m*n,m*(n-1):m*n] + sub4
#
# Lxy = Lxy + sub1 + sub2
#
#
# Dxx = Lxx.dot(g_vec) + math.pow(k,2)
# Dyy = Lyy.dot(g_vec) + math.pow(k,2)
#
# Dxy = 1/2*(Lxy.dot(g_vec) + Dxx + Dyy)

v = (np.array([1, 2, 3])).T
u = np.array([2, 1, 4]).T
print u.shape
print np.multiply(u,v)
print (u).dot((v.T))
print np.power(v,2)

#
# lam = 0.0001
# D = HS.makeDmatrix(g)
# L = HS.makeLmatrix(m,n)
# M = (D.T).dot(D)
# # RHS
# b = -(D.T).dot(c)
#
# w = np.ones(2*m*n)
#
# # Directions nomal to edges and parallel to edges
# [r1,r2] = cg.regTensorEigenvector(g)
#
#
# # Flow derivatives
# grad_w = L.dot(w)
# # Diffusion matrix
# Dif = cg.makeDiffusionMatrix2(r1,r2,grad_w,m,n,lam)
# # Smoothness term, div(Dif*grad(w))
# V = (L.T).dot(Dif.dot(L))
#
# G = M + math.pow(regu,-2)*V
#
# P = mg.makeProlongationMatrix(m,n)
#
#
# GH = R*G*P
#
# r = b-G.dot(w)
# e = spsolve(G,r)
#
# print e.shape
# print R.shape
#
#
# rH = R*r
#
# eH = spsolve(GH,rH)
#
# print eH.shape

# a = np.array([1,2])
# A = np.kron(np.eye(3),a) + np.kron(np.diags(np.ones()))
# print A

# A = np.array([[1,2,3],[1,2,3]])
# a = np.reshape(A.T,[1,2*3])[0]
# print np.reshape(a,[3,2]).T
#
# #g = ndimage.filters.gaussian_filter(g1,0.5)
# g = g1
# gamma = 0.1
# k = 0.0001 # Small parameter to avoid singular matrices
# [Dx, Dy] = HS.forwardDifferenceImage(g)
# gx = np.reshape(Dx,[n,m]).T
# gy = np.reshape(Dy,[n,m]).T
# [Dxx,Dxy] = HS.forwardDifferenceImage(gx)
# [Dyx,Dyy] = HS.forwardDifferenceImage(gy)
#
# theta_0 = np.sqrt(np.power(Dx,2) + np.power(Dy,2) + k)
# theta_x = np.sqrt(np.power(Dxx,2) + np.power(Dxy,2) + k)
# theta_y = np.sqrt(np.power(Dyx,2) + np.power(Dyy,2) + k)
#
# sx = sparse.diags(np.divide(Dx,theta_0),0,format='csr')
# sy = sparse.diags(np.divide(Dy,theta_0),0,format='csr')
# d = sparse.hstack((sx,sy),format='csr').T
#
# sxx = sparse.diags(np.divide(Dxx,theta_x),0,format='csr')
# sxy = sparse.diags(np.divide(Dxy,theta_x),0,format='csr')
# dx = sparse.hstack((sxx,sxy),format='csr').T
#
# syx = sparse.diags(np.divide(Dyx,theta_y),0,format='csr')
# syy = sparse.diags(np.divide(Dyy,theta_y),0,format='csr')
# dy = sparse.hstack((syx,syy),format='csr').T
#
# R = d.dot(d.T) + gamma*(dx.dot(dx.T)+ dy.dot(dy.T))
# R11 = R[0:m*n,0:m*n]
# R12 = R[0:m*n,m*n:2*m*n]
# R21 = R[m*n:2*m*n,0:m*n]
# R22 = R[m*n:2*m*n,m*n:2*m*n]
# tmp = np.divide(np.sqrt(np.power(R11,2)-2*R11.dot(R22)+np.power(R22,2)+4*R12.dot(R21)),2*R21)
# r2 = sparse.hstack((np.divide(R11 + R22,2*R21) - tmp,sparse.eye(m*n)),format = 'csr').T
# # r1 is the eigenvector corresponding to the largest eigenvalue
# r1 = sparse.hstack((np.divide(R11 + R22,2*R21) + tmp - np.divide(R22,R21),sparse.eye(m*n)),format = 'csr').T
#
# print r1
# print r2
