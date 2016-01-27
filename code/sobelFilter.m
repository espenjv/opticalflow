function [Dx,Dy] = sobelFilter(g)
%sobelFilter Apply sobelfilter to an image
%   Detailed explanation goes here

    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = Sx';
    [m,n] = size(g);
    
    Dx = conv2(double(g),double(Sx));
    Dx = Dx(2:m+1,2:n+1);
    Dy = conv2(double(g),double(Sy));
    Dy = Dy(2:m+1,2:n+1);
end