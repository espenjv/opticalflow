function [Dx,Dy] = sobelFilter(g)
%sobelFilter Apply sobelfilter to an image
%   Detailed explanation goes here

    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = Sx';
    [m,n] = size(g);
    
    Dx = conv2(double(g),double(Sx),'same');
    Dy = conv2(double(g),double(Sy),'same');
    
    Dx = reshape(Dx,[m*n 1]);
    Dy = reshape(Dy,[m*n 1]);
    
end