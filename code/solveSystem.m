clear all
close all

path = 'C:\Users\Espen\opticalflow\HamburgTaxi\';
ext = '*.tif';

imCol = getImages(path,ext);

g1 = imCol{1};
g2 = imCol{2};

% g1 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000000.png');
% g2 = imread('/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/0000000001.png');

g = g1;

[m,n] = size(g1);

[Dx,Dy] = sobelFilter(g);
Dx = reshape(Dx,[m*n 1]);
Dy = reshape(Dy,[m*n 1]);

% Dx = forwardDifferenceImage(g);
% Dy = forwardDifferenceImage(g');

% Dx = backwardDifferenceImage(g);
% Dy = backwardDifferenceImage(g');


D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);

% figure
% imshow(reshape(Dx,[m n]))
% 
% figure
% imshow(reshape(Dy,[m n]))


c = timeDisc(g1,g2);
c = reshape(c,[m*n 1]);
L = forwardDifference(m,n);
reg = 0.01;



G = D'*D + reg^(-2)*(L'*L);

u = G\(-D'*c);

X = repmat(1:n,1,m)';
Y = repmat(1:m,n,1);
Y = Y(:);

% figure
% quiver(X,Y,u(1:m*n),u(m*n+1:end),0)

F = sqrt(u(1:m*n).^2+u(m*n+1:end).^2);
F = reshape(F,[m n]);

figure
imshow(F)
axis equal

% figure
% contourf(reshape(u(1:m*n),[m n]))
% colorbar
% 
% figure
% contourf(reshape(u(m*n+1:end),[m n]))
% colorbar