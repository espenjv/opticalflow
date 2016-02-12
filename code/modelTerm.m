function M = modelTerm(Dx,Dy,m,n)
%modelTerm Computes the model term from the method of Horn and Schunck.
%   The model term comes from Horn and Schunck, but is also used in the
%   method of Nagel and Enkelmann.

    D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
    D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);
    
    M = D'*D;

end

