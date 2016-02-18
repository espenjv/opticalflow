close all
clear all

method = 'HS';
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

flow_col = cell(4,1);
reg_vec = [0.01, 0.007,0.005,0.003];
figure
title('HS for different regularization parameters')
for i = 1:4
    u = findFlow(g1,g2,method,gradient,reg_vec(i));
    flow = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
    subplot(2,2,i)
    imshow(flow)
    title(['R = ', num2str(reg_vec(i))])
end




