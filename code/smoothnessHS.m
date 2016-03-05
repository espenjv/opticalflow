function V = smoothnessHS(m,n)
%smoothnessHS Computes smoothness term by Horn and Schunck
%   Uses forward difference to compute the first derivatives

    [Lx,Ly] = forwardDifference(m,n);
    
    
    
    L1 = sparse(2*m*n,m*n);
    L1(1:m*n,1:m*n) = Lx;
    L1(m*n+1:2*m*n,1:m*n) = Ly;
    L = kron(eye(2),L1);
    
    
    V = L'*L;



end

