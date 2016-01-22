function [Lx,Ly] = flowLaplace(m,n)
%flowLaplace Computes the Laplace matrices
%   Detailed explanation goes here
    
    Ly = diag(ones(m-1,1),-1) -2*diag(ones(m,1)) + diag(ones(m-1,1),1);
    Lx = diag(ones(n-1,1),-1) -2*diag(ones(n,1)) + diag(ones(n-1,1),1);

end

