function M = modelTerm(Dx,Dy,m,n)
%modelTerm Summary of this function goes here
%   Detailed explanation goes here

    D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
    D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);
    
    M = D'*D;

end
