    %% mnist_check.m  (v2)  —  Έλεγχος & προετοιμασία MNIST
    rootFolder = 'C:\Users\AndreasK\Documents\EX_2\mnist';
    addpath(rootFolder);                                   % βλέπει τα .m αν υπάρχουν
    fprintf('Χρησιμοποιώ φάκελο: %s\n',rootFolder);
    
    files = { 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', ...
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte' };
    
    %% --- 1. Έλεγχος αρχείων & magic numbers ---------------------------------
    for k = 1:numel(files)
        f = fullfile(rootFolder,files{k});
        assert(isfile(f),'Λείπει: %s',f);
        fid = fopen(f,'rb');
        magic = fread(fid,1,'uint32','ieee-be');
        n     = fread(fid,1,'uint32','ieee-be');
        if contains(files{k},'images')
            r = fread(fid,1,'uint32','ieee-be'); c = fread(fid,1,'uint32','ieee-be');
            fprintf('%s ➜ magic=%d  N=%d  %dx%d\n',files{k},magic,n,r,c);
            assert(magic==2051,'Magic ≠2051!');
        else
            fprintf('%s ➜ magic=%d  N=%d\n',files{k},magic,n);
            assert(magic==2049,'Magic ≠2049!');
        end
        fclose(fid);
    end
    
    %% --- 2. Φόρτωση σε μνήμη -------------------------------------------------
    mnist.trainImages = readImages(fullfile(rootFolder,files{1}));
    mnist.trainLabels = readLabels(fullfile(rootFolder,files{2}));
    mnist.testImages  = readImages(fullfile(rootFolder,files{3}));
    mnist.testLabels  = readLabels(fullfile(rootFolder,files{4}));
    fprintf('\nΔομή mnist έτοιμη: train=%d  test=%d\n', ...
            numel(mnist.trainLabels), numel(mnist.testLabels));
    
    %% --- 3. Οπτική επαλήθευση (25 τυχαία) 
    idx = randperm(numel(mnist.trainLabels),25);
    figure('Name','Τυχαία ψηφία MNIST');
    tiledlayout(5,5,'Padding','compact');
    for i = 1:25
        nexttile
        imshow(mnist.trainImages(:,:,idx(i)),[])
        title(num2str(mnist.trainLabels(idx(i))));
    end
    
    %% προετοιμήστε 4-D πίνακα & κατηγορίες 
    XTrain = single(reshape(mnist.trainImages,28,28,1,[])) / 255;   % H×W×C×N
    YTrain = categorical(mnist.trainLabels);
    XTest  = single(reshape(mnist.testImages ,28,28,1,[])) / 255;
    YTest  = categorical(mnist.testLabels);
    
    % δείγμα κλήσης trainNetwork χωρίς datastore:
    % layers = [...]; options = trainingOptions(...);
    % net = trainNetwork(XTrain, YTrain, layers, options);
    
    %%  datastore με combine (εικόνες + ετικέτες) --------------
    dsX = arrayDatastore(XTrain,'IterationDimension',4);
    dsY = arrayDatastore(YTrain);
    dsTrain = combine(dsX,dsY);          % returns {imagesBatch, labelsBatch}
    
    augTrain = transform(dsTrain, @(data) deal(data{1},data{2}));  % τυπική τυλιχτή
    disp('Έτοιμο CombinedDatastore (augTrain) για trainNetwork');
    
    %% Helper functions 
    function imgs = readImages(fname)
    fid = fopen(fname,'rb'); fread(fid,1,'uint32','ieee-be');       % magic
    n  = fread(fid,1,'uint32','ieee-be');
    r  = fread(fid,1,'uint32','ieee-be'); c = fread(fid,1,'uint32','ieee-be');
    imgs = fread(fid,r*c*n,'uint8=>uint8'); fclose(fid);
    imgs = permute(reshape(imgs,[c,r,n]),[2 1 3]);
    end
    
    function lbl = readLabels(fname)
    fid = fopen(fname,'rb'); fread(fid,1,'uint32','ieee-be');       % magic
    n = fread(fid,1,'uint32','ieee-be'); lbl = fread(fid,n,'uint8=>uint8');
    fclose(fid);
    end

Xtrain = mnist.trainImages;            % 28×28×60000      (uint8 0-255)
Ytrain = mnist.trainLabels;            % 1×60000          (uint8)
Xtest  = mnist.testImages;             % 28×28×10000
Ytest  = mnist.testLabels;             % 1×10000

save(fullfile('C:\Users\AndreasK\Documents\EX_2\mnist', ...
              'mnistData.mat'), ...
     'Xtrain','Ytrain','Xtest','Ytest','-v7.3');
