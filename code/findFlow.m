function w = findFlow(g1,g2,method,gradient,regu,kappa)

    if nargin < 6
        kappa = 0.01;
    end

    g = g1;
    [m,n] = size(g);

    if strcmp(gradient,'sobel')
        [Dx,Dy] = sobelFilter(g);
    elseif strcmp(gradient,'forward')
        Dx = forwardDifferenceImage(g);
        Dy = forwardDifferenceImage(g');
    elseif strcmp(gradient,'backward')
        Dx = backwardDifferenceImage(g);
        Dy = backwardDifferenceImage(g');
    end

    D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
    D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);

    c = timeDisc(g1,g2);
    c = reshape(c,[m*n 1]);

    if strcmp(method,'HS')
        % Uses the method of Horn and Schunck
        M = D'*D;
        V = smoothnessHS(m,n);
        G = M + regu^(-2)*V;
        w = G\(-D'*c);
    elseif strcmp(method,'NE')
        % Uses the method of Nagel and Enkelmann
        M = D'*D;
        V = smoothnessNE(Dx,Dy,m,n,kappa);
        G = M + regu^(-2)*V;
        w = G\(-D'*c);
    elseif strcmp(method,'SH')
        param = 1;
        w = flowdrivenSH(Dx,Dy,c,m,n,param,'cohen');
    end


end