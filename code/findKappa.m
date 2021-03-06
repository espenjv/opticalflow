close all
clear all

method = 'NE';
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

[m,n] = size(g1);

reg = 0.003;
flow_col = cell(4,1);
kappa_vec = [1, 0.8,0.5,0.2];
figure
for i = 1:4
    u = findFlow(g1,g2,method,gradient,reg,kappa_vec(i));
    flow = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
    subplot(2,2,i)
    imshow(flow)
    title(['\kappa = ', num2str(kappa_vec(i))])
end
suptitle('Flow field for  \kappa')

figure
u = findFlow(g1,g2,method,gradient,reg,0.8);
flow = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
subplot(1,2,1)
imshow(flow)
title(['NE: \kappa = ', num2str(0.8), ', \sigma =', num2str(reg)])
u = findFlow(g1,g2,'HS',gradient,reg);
flow = display.computeColor(reshape(u(1:m*n),[m n]),reshape(u(m*n+1:end),[m n]));
subplot(1,2,2)
imshow(flow)
title(['HS: \sigma = ', num2str(reg)])
