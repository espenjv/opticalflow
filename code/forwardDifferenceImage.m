function D = forwardDifferenceImage(g)
%forwardDifferenceImage Computes approximation of the image gradient using
%forward difference
%   Detailed explanation goes here

    [m,n] = size(g);

    g_vec = reshape(double(g),[m*n 1]);

    L = sparse(1:m*n,1:m*n,-ones(m*n,1),m*n,m*n) + sparse(1:m*n-1,2:m*n,ones(m*n-1,1),m*n,m*n);
    
    D = L*g_vec;


end

