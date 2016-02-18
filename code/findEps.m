close all
clear all

method = 'SH';
image_set = 'taxi';
load_set = 0;
gradient = 'sobel';

if strcmp(image_set,'taxi')
    if load_set
        path = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/';
        ext = '*.tif';
        imCol = getImages(path,ext);
        g1 = imCol{1};
        g2 = imCol{2};
    else
        g1 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-00.tif');
        g2 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-01.tif');
    end
elseif strcmp(image_set,'bicycle')
    if load_set
        path = '/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/';
        ext = '*.png';
        imCol = getImages(path,ext);
        g1 = imCol{1};
        g2 = imCol{2};
    else
        g1 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000000.png');
        g2 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000001.png');
    end
end

g = g1;
[m,n] = size(g);

[Dx,Dy] = sobelFilter(g);
c = timeDisc(g1,g2);
c = reshape(c,[m*n 1]);

reg = 0.007;
eps_vec = [1, 0.7,0.5,0.1];
figure
title('SH for chosen alpha parameter and different epsilon')
for i = 1:4
    u = flowdrivenSH(Dx,Dy,c,m,n,eps_vec(i),'cohen');
    flow = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
    subplot(2,2,i)
    imshow(flow)
    title(['eps = ', num2str(eps_vec(i))])
end

