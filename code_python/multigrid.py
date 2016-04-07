import numpy as np
from scipy import sparse

def makeRestrictionMatrix(m,n):

    R_sub1 = sparse.kron(sparse.eye(m/2),[1.0/16, 1.0/8]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/16,0])
    R_sub2 = sparse.kron(sparse.eye(m/2),[1.0/8, 1.0/4]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/8,0])
    R1 = sparse.hstack((R_sub1,R_sub2))
    R = sparse.kron(sparse.eye(n/2),R1)+ sparse.kron(sparse.diags(np.ones(n/2-1),1),sparse.hstack((R_sub1,np.zeros((m/2,m)))))
    R = sparse.kron(sparse.eye(2),R,format = 'csr')
    return R
def makeProlongationMatrix(m,n):
    P_sub1 = sparse.kron(sparse.eye(m/2),[1.0/4, 1.0/2]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/4,0])
    P_sub2 = sparse.kron(sparse.eye(m/2),[1.0/4, 1.0]) + sparse.kron(sparse.diags(np.ones(m/2-1),1),[1.0/4,0])
    P1 = sparse.hstack((P_sub1,P_sub2))
    P = sparse.kron(sparse.eye(n/2),P1)+ sparse.kron(sparse.diags(np.ones(n/2-1),1),sparse.hstack((P_sub1,np.zeros((m/2,m)))))
    P = sparse.kron(sparse.eye(2),P,format = 'csr')
    return P
