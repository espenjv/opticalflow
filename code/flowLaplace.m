function L = flowLaplace(m,n)
%flowLaplace Computes the Laplace matrices
%   Detailed explanation goes here
    
    L = diag(ones(m+n-1,1),-1) -2*diag(ones(m+n,1)) + diag(ones(m+n-1,1),1);
    L(m+1,m) = 0;
    L(m,m+1) = 0;

end

