clear all
close all

g1 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-00.tif');
g2 = imread('/home/shomec/e/espenjv/Semester Project/HamburgTaxi/taxi-01.tif');
[m,n] = size(g1);
[Dx,Dy] = sobelFilter(g1);
size(Dx)

Dx = reshape(Dx,[m*n 1]);
Dy = reshape(Dy,[m*n 1]);
D = sparse(1:m*n,1:m*n,Dx,m*n,2*m*n);
D(:,m*n+1:2*m*n) = sparse(1:m*n,1:m*n,Dy,m*n,m*n);
c = timeDisc(g1,g2);
c = reshape(c,[m*n 1]);
L = forwardDifference(m,n);
reg = 1;

G = D'*D + reg^(-2)*L'*L;

u = G\(-D'*c);

X = repmat(1:n,1,m)';
Y = repmat(1:m,n,1);
Y = Y(:);

figure
quiver(X,Y,u(1:m*n),u(m*n+1:end))