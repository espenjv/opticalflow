function [Gx,Gy] = sobelFilter(im)
%sobelFilter Apply sobelfilter to an image
%   Detailed explanation goes here

    [m,n] = size(im);

    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = Sx';
    
    Gx = conv2(double(im),double(Sx));
    Gy = conv2(double(im),double(Sy));
end