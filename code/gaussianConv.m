function f_conv = gaussianConv(f)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    g = 1/273*[1 4 7 4 1; 4 16 26 16 4; 7 26 41 26 7; 4 16 26 16 4; 1 4 7 4 1];
    
    f_conv = conv2(double(f),double(g));
    
    [m,n] = size(f_conv);
    
    f_conv = f_conv(3:m-2,3:n-2);

end

