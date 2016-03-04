from scipy import misc, ndimage,sparse



def findFlow(g1,g2,method,gradient,regu,kappa=0.01):
    g = g1
    [m,n] = g.shape

    if gradient is 'sobel':
        Dx = ndimage.filters.sobel(g1,0,mode='constant')
        Dx = misc.imresize(Dx,[m*n,1])
        Dy = ndimage.filters.sobel(g1,1,mode='constant')
        Dy = misc.imresize(Dy,[m*n,1])

    D = sparse.bmat(diag(Dx),diag(Dy))
