function [Lx,Ly] = forwardDifference(m,n)
%forwardDifference Forms the forwardDifference matrix
%   Detailed explanation goes here
    
    Lx = sparse(1:m*n,1:m*n,-ones(m*n,1),m*n,m*n) + sparse(1:m*(n-1),m+1:m*n,ones((n-1)*m,1),m*n,m*n);
    Ly = kron(eye(n),sparse(1:m,1:m,-ones(m,1),m,m) + sparse(1:m-1,2:m,ones(m-1,1),m,m));
    
end

