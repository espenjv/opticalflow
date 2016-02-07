function V = smoothnessNE(Dx,Dy,m,n)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    grad = sqrt(Dx.^2+Dy.^2);
    sx = Dx./grad;
    sy = Dy./grad;
  
    
    P1 = sparse(1:m*n,1:m*n,sx,2*m*n,2*m*n);
    P1(1:m*n,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,sy,m*n,m*n);
    P1(m*n+1:2*m*n,1:m*n) = sparse(1:m*n,1:m*n,-sy,m*n,m*n);
    P1(m*n+1:2*m*n,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,sx,m*n,m*n);
    P = kron(eye(2),P1);
    
    G1 = sparse(1:m*n,1:m*n,grad,m*n,2*m*n);
    G1(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,ones(m*n,1),m*n,m*n);
    G = [G1 G1];
    
    [Lx,Ly] = forwardDifference(m,n);
    
    L1 = sparse(2*m*n,m*n);
    L1(1:m*n,1:m*n) = Lx;
    L1(m*n+1:2*m*n,1:m*n) = Ly;
    L = kron(eye(2),L1);
    
    
    V = (P*L)'*(P*L);
    
    V = L'*L;
    
    V = (V>10^-15).*V;
    

end

