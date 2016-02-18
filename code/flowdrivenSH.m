function u = flowdrivenSH(Dx,Dy,c,m,n,param,penal)
%flowdrivenSH Computes flow using the method of Shulman and Herv
%   Flow driven regularization with and subquadratic penaliser function for
%   the smoothness term (convex)
    regu = 0.007;

    D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
    D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);
    M = D'*D;
    I = sparse(1:m*n,1:m*n,ones(m*n,1),m*n,2*m*n);
    I(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,ones(m*n,1),m*n,m*n);
    I = [I, I];
    
    [Lx,Ly] = forwardDifference(m,n);
    
    L1 = sparse(2*m*n,m*n);
    L1(1:m*n,1:m*n) = Lx;
    L1(m*n+1:2*m*n,1:m*n) = Ly;
    L = kron(eye(2),L1);
    
    
    u = zeros(2*m*n,1);
    
    if strcmp(penal,'cohen')
        
        eps = param;
        
        V1 = spdiags(ones(m*n,1)./(2*sqrt(I*(L*u).^2+eps^2)),0,m*n,m*n);
        V = L'*kron(eye(4),V1)*L;

        del_u = 1;


        while max(del_u) > 10^(-2)

            G = M + regu^(-2)*V;
            u_new = G\(-D'*c);
            V1 = spdiags(ones(m*n,1)./(2*sqrt(I*(L*u_new).^2+eps^2)),0,m*n,m*n);
            V = L'*kron(eye(4),V1)*L;

            del_u = u_new - u;
            u = u_new;

        end
    elseif strcmp(penal,'PM')
        lambda = param;
        
        V1 = spdiags(ones(m*n,1)./((I*(L*u).^2)/lambda+1),0,m*n,m*n);
        V = L'*kron(eye(4),V1)*L;

        del_u = 1;

        while max(del_u) > 10^(-2)

            G = M + regu^(-2)*V;
            u_new = G\(-D'*c);
            V1 = spdiags(ones(m*n,1)./((I*(L*u_new).^2)/lambda+1),0,m*n,m*n);
            V = L'*kron(eye(4),V1)*L;

            del_u = u_new - u;
            u = u_new;

        end
    end
end

