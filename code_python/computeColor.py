import makeColorwheel as mC
import numpy as np
import math


def computeColor(u,v,m,n):

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = mC.makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.add(np.power(u,2),np.power(v,2)))
    a = np.arctan2(-v, -u)/math.pi;
    fk = (a+1) /2 * (ncols-1) + 1;  #-1~1 maped to 1~ncols
    k0 = np.floor(fk);                 # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1==ncols+1] = 0;
    f = (fk - k0);
    k0 = np.array(k0,dtype=np.int)
    k1 = np.array(k1,dtype=np.int)

    i = 0
    img = np.zeros([m,n,3])
    for i in np.arange(colorwheel.shape[1]):
        tmp = colorwheel[:,i];
        col0 = tmp[k0-1]/255;
        col1 = tmp[k1-1]/255;
        col = np.multiply(1-f,col0) + np.multiply(f,col1);

        idx = rad <= 1;
        col[idx] = 1-np.multiply(rad[idx],(1-col[idx]));    # increase saturation with radius

        col[~idx] = col[~idx]*0.75;             # out of range


        img[:,:, i] = np.array(np.floor(255*np.multiply(col,(1-nanIdx))),dtype=np.float);


    return img
