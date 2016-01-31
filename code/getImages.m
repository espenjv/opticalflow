function imCol = getImages(path,ext)
%getImages Loads images from a given path with a given extension. Returns a
%cell array with the images in matrix form
%   Path - string with the name of the path
        % Example: path = 'C:\Users\Espen\opticalflow\HamburgTaxi\'
%   ext - String with the extension of the images
        % Exampe: ext = '*.tif'

    images = dir([path ext]);
    n = length(images);
    imCol = cell(n,1);
    
    for i = 1:n
        imCol{i} = imread([path images(i).name]);
    end

end

