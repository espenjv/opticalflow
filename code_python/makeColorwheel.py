import numpy as np

def makeColorwheel():
    RY = 15;
    YG = 6;
    GC = 4;
    CB = 11;
    BM = 13;
    MR = 6;

    ncols = RY + YG + GC + CB + BM + MR;

    colorwheel = np.zeros([ncols, 3]); # r g b

    col = 0;
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(RY)/RY).T
    col = col+RY;

    #YG
    colorwheel[col+np.arange(YG), 0] = 255 - np.floor(255*np.arange(YG)/YG).T;
    colorwheel[col+np.arange(YG), 1] = 255;
    col = col+YG;

    #GC
    colorwheel[col+np.arange(GC), 1] = 255;
    colorwheel[col+np.arange(GC), 2] = np.floor(255*np.arange(GC)/GC).T;
    col = col+GC;


    #CB
    colorwheel[col+np.arange(CB), 1] = 255 - np.floor(255*np.arange(CB)/CB).T;
    colorwheel[col+np.arange(CB), 2] = 255;
    col = col+CB;

    #BM
    colorwheel[col+np.arange(BM), 2] = 255;
    colorwheel[col+np.arange(BM), 0] = np.floor(255*np.arange(BM)/BM).T;
    col = col+BM;

    #MR
    colorwheel[col+np.arange(MR), 2] = 255 - np.floor(255*np.arange(MR)/MR).T;
    colorwheel[col+np.arange(MR), 0] = 255;

    return colorwheel
