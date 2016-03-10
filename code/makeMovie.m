function  makeMovie(method,image_set,outputfile)

    if strcmp(image_set,'taxi')
        path = '/home/shomec/e/espenjv/Semester Project/HamburgTaxi/';
        ext = '*.tif';
        imCol = getImages(path,ext);
    elseif strcmp(image_set,'bicycle')
        path = '/home/shomec/e/espenjv/Semester Project/2011_09_26/2011_09_26_drive_0060_extract/image_00/data/';
        ext = '*.png';
        imCol = getImages(path,ext);
    end
    
    [m,n] = size(imCol{1});
    
    k = length(imCol);
    flow_col = cell(k,1);
    fr1 = cat(2,zeros(m,n,3),cat(3,imCol{1},imCol{1},imCol{1}));
    vid = VideoWriter(outputfile);
    vid.FrameRate = 15;
    open(vid)
    writeVideo(vid,fr1)
    for i = 1:k-1
        g1 = imCol{i};
        g2 = imCol{i+1};
        w = findFlow(g1,g2,method,'sobel',0.003);
        u = w(1:m*n);
        v = w(m*n+1:end);
        flow_image = display.computeColor(reshape(u,[m n]),reshape(v,[m n]));
        flow_col{i} = flow_image;
        fr = cat(2,flow_image,cat(3,g2,g2,g2));
        writeVideo(vid,fr)
    end
    close(vid)
%     save('BicycleFlow',flow_col)
end