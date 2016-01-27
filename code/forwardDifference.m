function L = forwardDifference(m,n)
%forwardDifference Forms the forwardDifference matrix
%   Detailed explanation goes here

    L = sparse(1:2*m*n,1:2*m*n,-ones(2*m*n,1),2*m*n,2*m*n) + sparse(1:2*m*n-1,2:2*m*n,ones(2*m*n-1,1),2*m*n,2*m*n);
    L(m*n,m*n+1) = 0;

end

