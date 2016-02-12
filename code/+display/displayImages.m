function [] = displayImages(g1,g2)
    figure
    subplot(1,2,1)
    imshow(g1)
    title('Image 1')

    subplot(1,2,2)
    imshow(g2)
    title('Image 2')
end