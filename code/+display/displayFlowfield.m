function [] = displayFlowfield(u,m,n)
    

    F = sqrt(u(1:m*n).^2+u(m*n+1:end).^2);
    F = reshape(F,[m n]);
    flow_image = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
    
    figure

    subplot(1,2,1)
    imshow(F)
    title('Flow vector absolute values')

    subplot(1,2,2)
    imshow(flow_image)
    title('Flow vector')
end