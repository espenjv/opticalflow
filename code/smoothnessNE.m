function V = smoothnessNE(Dx,Dy,m,n,kappa)
%smoothnessNE Computes the smoothness
%   Computes the smoothness term using the method of Nagel and Enkelmann.
%   The method uses a projection matrix P to give different wheights to the
%   components of the flow deriatives in the direction of the image
%   gradient and the orthogonal direction. Kappa is a regularization
%   parameter > 0.

    grad = Dx.^2+Dy.^2;
    denom = grad + kappa^2;
    sx = Dx.^2./denom;
    sy = Dy.^2./denom;
    sxy = Dx.*Dy./denom;
  
    P1 = sparse(1:m*n,1:m*n,sy+kappa^2,2*m*n,2*m*n);
    P1(1:m*n,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,-sxy,m*n,m*n);
    P1(m*n+1:2*m*n,1:m*n) = sparse(1:m*n,1:m*n,-sxy,m*n,m*n);
    P1(m*n+1:2*m*n,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,sx+kappa^2,m*n,m*n);
    P = kron(eye(2),P1);
    
    [Lx,Ly] = forwardDifference(m,n);
    
    L1 = sparse(2*m*n,m*n);
    L1(1:m*n,1:m*n) = Lx;
    L1(m*n+1:2*m*n,1:m*n) = Ly;
    L = kron(eye(2),L1);
    
    V = L'*P*L; 

end

