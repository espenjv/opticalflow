close all
clear all


method = 'HS';
image_set = 'bicycle';
load_set = 0;
gradient = 'sobel';

if strcmp(image_set,'taxi')
    if load_set
        path = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/';
        ext = '*.tif';
        imCol = getImages(path,ext);
        g1 = imCol{1};
        g2 = imCol{2};
    else
        g1 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-00.tif');
        g2 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-01.tif');
    end
elseif strcmp(image_set,'bicycle')
    if load_set
        path = '/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/';
        ext = '*.png';
        imCol = getImages(path,ext);
        g1 = imCol{1};
        g2 = imCol{2};
    else
        g1 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000000.png');
        g2 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000001.png');
    end
end
       

g = g1;
[m,n] = size(g);

g = gaussianConv(g);

if strcmp(gradient,'sobel')
    [Dx,Dy] = sobelFilter(g);
elseif strcmp(gradient,'forward')
    Dx = forwardDifferenceImage(g);
    Dy = forwardDifferenceImage(g');
elseif strcmp(gradient,'backward')
    Dx = backwardDifferenceImage(g);
    Dy = backwardDifferenceImage(g');
end

D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);

c = timeDisc(g1,g2);
c = reshape(c,[m*n 1]);

if strcmp(method,'HS')
    % Uses the method of Horn and Schunck
    c = timeDisc(g1,g2);
    c = reshape(c,[m*n 1]);
    M = D'*D;
    V = smoothnessHS(m,n);
    regu = 0.003;
    G = M + regu^(-2)*V;
    u = G\(-D'*c);
elseif strcmp(method,'NE')
    % Uses the method of Nagel and Enkelmann
    c = timeDisc(g1,g2);
    c = reshape(c,[m*n 1]);
    M = D'*D;
    kappa = 10;
    V = smoothnessNE(Dx,Dy,m,n,kappa);
    regu = 0.007;
    G = M + regu^(-2)*V;
    u = G\(-D'*c);
elseif strcmp(method,'SH')
    param = 0.1;
    c = timeDisc(g1,g2);
    c = reshape(c,[m*n 1]);
    u = flowdrivenSH(Dx,Dy,c,m,n,param,'PM');
end
display.displayImages(g1,g2)

display.displayFlowfield(u,m,n)

