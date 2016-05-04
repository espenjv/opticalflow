def makeLmatrix(m,n):
    # Forms the L matrix used in the flow derivative approximation.
    # A matrix multipication with Lx and Ly gives the approximation of the
    # derivatives in x- and y-direction assuming neumann boundary conditions.
    # Parameters: m: number of rows in the image
    #             n: number of columns in the image
    # Returns: L: 2-dimensional array of shape 4mn x 2mn. The matrix
    #           multiplication grad_w= L[u,v].T gives a
    #           4mn vector grad_w = [u_x,u_y,v_x,v_y].T
    k = 0.00001
    # Lx = sparse.kron(sparse.eye(n),sparse.hstack((sparse.eye(m),np.zeros((m,m)))),format = 'csr') + sparse.kron(sparse.eye(n),sparse.hstack((np.zeros((m,m)),-sparse.eye(m))), format = 'csr')
    Lx = sparse.diags([-np.ones(m*n),np.ones((n-1)*m)],[0,m],format = 'lil')
    # Lx[:m,:2*m] = np.zeros(m,2*m)
    # Lx[m*(n-1):m*n,m*(n-1):m*n] = np.zeros((m,m))
    Ly1 = sparse.diags([-np.ones(m), np.ones(m-1)],[0,1],format = 'lil')
    # Ly[0,:2] = [0,0]
    # Ly1[m-1,:] = np.zeros(m)
    Ly = sparse.kron(sparse.eye(n),Ly1,format = 'csr')
    L = sparse.kron(sparse.eye(2),sparse.vstack((Lx,Ly)),format = 'csr')
    return L

def neumann_boundary(G,b,m,n):
    neumann_x = sparse.hstack((sparse.diags(-np.ones(m),0),sparse.diags(np.ones(m),0)))
    neumann_y = np.zeros((m,m))
    neumann_y[0,:2] = [-1,1]
    neumann_y[m-1,m-2:m] = [-1,1]
    neumann_y = sparse.kron(sparse.eye(n-2),neumann_y)
    neumann = sparse.block_diag((sparse.diags(-np.ones(m),0),neumann_y,sparse.diags(-np.ones(m),0)),format='lil')
    neumann[:m,m:2*m] = sparse.diags(np.ones(m),0)
    neumann[m*(n-1):m*n,(n-2)*m:(n-1)*m] = sparse.diags(np.ones(m),0)
    elim_y = np.ones(m)
    elim_y[0] = 0
    elim_y[m-1] = 0
    elimination_vector = np.hstack((np.hstack((np.zeros(m),np.kron(np.ones((n-2)),elim_y))),np.zeros(m)))
    elimination_vector = np.kron(np.ones(2),elimination_vector)
    G = sparse.diags(elimination_vector,0).dot(G) + sparse.kron(sparse.eye(2),neumann)
    b = elimination_vector*b
    return G,b


## Methods for solving Euler-Lagrange system

def smoothnessHS(m,n):
    # Computes the smoothness term of Horn and Schunck.
    # Parameters: m: number of rows in the image
    #             n: number of columns in the image
    # Returns: V: 2mn x 2mn array

    L = makeLmatrix(m,n)
    V = (L.T).dot(L)
    return V
