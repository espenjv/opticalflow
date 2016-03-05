function [] = displayFlowfield(w,m,n)
    
    u = w(1:m*n);
    v = w(m*n+1:end);
    temp = u.^2+v.^2;
    F = sqrt(u.^2+v.^2);
    F = reshape(F,[m n]);
    flow_image = display.computeColor(reshape(u,[m n]),reshape(v,[m n]));
    
    figure

    imshow(F)
    title('Flow vector absolute values')
    
    figure
    imshow(flow_image)
    title('Flow vector')
end