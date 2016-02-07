clear all
close all

% path = '/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/';
% % path = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/';
% %path = 'C:\Users\Espen\opticalflow\HamburgTaxi';
% % ext = '*.tif';
% ext = '*.png';
% 
% imCol = getImages(path,ext);
% 
% g1 = imCol{1};
% g2 = imCol{2};

g1 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000000.png');
% g2 = g1;
g2 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000001.png');

g = g1;



% Dx = forwardDifferenceImage(g);
% Dy = forwardDifferenceImage(g');

% Dx = backwardDifferenceImage(g);
% Dy = backwardDifferenceImage(g');

[m,n] = size(g);

[Dx,Dy] = sobelFilter(g);
D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);

% Model Term Horn and Schunck
M = D'*D;


% figure
% imshow(reshape(Dx,[m n]))
% 
% figure
% imshow(reshape(Dy,[m n]))


c = timeDisc(g1,g2);
c = reshape(c,[m*n 1]);

% Smoothness term Horn and Schunck
% V = smoothnessHS(m,n);

% Smoothness term Nagel and Enkelmann
V = smoothnessNE(Dx,Dy,m,n);

regu = 0.01;

% u1_s1 = (D*Lxy*u(1:m*n)).^2;
% u1_s2 = (Dt*Lxy*u(1:m*n)).^2;
% u2_s1 = (D*Lxy*u(m*n+1:end)).^2;
% u2_s2 = (Dt*Lxy*u(m*n+1:end)).^2;
% 
% V = u1_s1 + u1_s2 + u2_s1 + u2_s1;

G = M + regu^(-2)*V;

u = G\(-D'*c);

X = repmat(1:n,1,m)';
Y = repmat(1:m,n,1);
Y = Y(:);

% figure
% quiver(X,Y,u(1:m*n),u(m*n+1:end),0)

F = sqrt(u(1:m*n).^2+u(m*n+1:end).^2);
F = reshape(F,[m n]);

figure
subplot(2,2,1)
imshow(g1)
title('Image 1')

subplot(2,2,2)
imshow(g2)
title('Image 2')

subplot(2,2,[3,4])
imshow(F)
title('Flow vector values')

% figure
% contourf(reshape(u(1:m*n),[m n]))
% colorbar
% 
% figure
% contourf(reshape(u(m*n+1:end),[m n]))
% colorbar