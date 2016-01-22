function [Dx,Dy] = sobelFilter(g)
%sobelFilter Apply sobelfilter to an image
%   Detailed explanation goes here

    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = Sx';
    
    Dx = conv2(double(g),double(Sx));
    Dy = conv2(double(g),double(Sy));
end