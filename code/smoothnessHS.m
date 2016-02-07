function V = smoothnessHS(m,n)
%smoothnessHS Computes smoothness term by Horn and Schunck
%   Detailed explanation goes here

    [Lx,Ly] = forwardDifference(m,n);
    
    L1 = sparse(2*m*n,m*n);
    L1(1:m*n,1:m*n) = Lx;
    L1(m*n+1:2*m*n,1:m*n) = Ly;
    L = kron(eye(2),L1);
    
    V = L'*L;



end

