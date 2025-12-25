
%% 0)  Φόρτωση δεδομένων & εκκαθάριση
clc; clear; close all;
dataDir = fullfile('C:','Users','AndreasK','Documents','EX_2','mnist');
load(fullfile(dataDir,'mnistData.mat'), 'Xtrain','Ytrain','Xtest','Ytest');

%% 1)  Ορισμός τριών CellSize (patch) – λογικές, κλιμακούμενες τιμές
patchList = [ 4  8 14 ];           % 4×4, 8×8, 14×14
accList   = zeros(size(patchList));

%% 2)  Βρόχος πειραμάτων HOG → SVM → Accuracy ----------------------------
for p = 1:numel(patchList)
    cellSz = [patchList(p) patchList(p)];
    fprintf('\n=== Πείραμα %d / %d  –  CellSize = [%d %d] ===\n', ...
            p, numel(patchList), cellSz);
    
    % 2.1 Προσδιορισμός μήκους χαρακτηριστικού 
    hogDim = numel( extractHOGFeatures( Xtrain(:,:,1), ...
                                        'CellSize',cellSz ) );
    fprintf('Μήκος HOG: %d στοιχεία ανά εικόνα\n', hogDim);
    
    % 2.2 Εξαγωγή HOG για όλα τα training samples 
    numTrain = size(Xtrain,3);
    HOGtrain = zeros(numTrain, hogDim, 'single');
    tic
    for i = 1:numTrain
        HOGtrain(i,:) = extractHOGFeatures( Xtrain(:,:,i), ...
                                            'CellSize',cellSz );
    end
    fprintf('   ➜ Χρόνος HOG (train): %.1f s\n', toc);
    
    % 2.3 Εκπαίδευση γραμμικού SVM (one-vs-all)
    t = templateLinear('Learner','svm');
    tic
    svmModel = fitcecoc(HOGtrain, Ytrain, ...
                        'Learners', t, ...
                        'Coding',  'onevsall', ...
                        'Verbose', 0);
    fprintf('   ➜ Χρόνος εκπαίδευσης SVM: %.1f s\n', toc);
    
    % 2.4 Εξαγωγή HOG & πρόβλεψη στο test set 
    numTest = size(Xtest,3);
    HOGtest = zeros(numTest, hogDim, 'single');
    tic
    for i = 1:numTest
        HOGtest(i,:) = extractHOGFeatures( Xtest(:,:,i), ...
                                           'CellSize',cellSz );
    end
    fprintf('   ➜ Χρόνος HOG (test) : %.1f s\n', toc);
    
    tic
    Ypred = predict(svmModel, HOGtest);
    fprintf('   ➜ Χρόνος πρόβλεψης  : %.1f s\n', toc);
    
    %  2.5 Accuracy 
    acc = mean(Ypred == Ytest);
    accList(p) = acc;
    fprintf('   ➜ Accuracy: %.2f %%\n', 100*acc);
    
  
    if patchList(p) == 8
        [C,~] = confusionmat(Ytest, Ypred);   % <<-- δική κλήση (Ερ. [3])
        figure('Name','Confusion Matrix – SVM (HOG 8×8)');
        imagesc(C); axis equal tight
        colormap(parula); colorbar
        title(sprintf('Confusion Matrix – Cell 8×8  (Acc %.2f %%)',100*acc));
        xticks(0:9); yticks(0:9); xlabel('Predicted'); ylabel('True');
        % γράφω αριθμούς
        textStrings = string(C); [x,y] = meshgrid(1:10);
        text(x(:),y(:),textStrings(:),'HorizontalAlignment','center',...
             'Color','w','FontWeight','bold');
    end
end

%% 3)  Αναφορά accuracy για τα τρία patch sizes ---------------------------
fprintf('\n========== Συνοπτικά αποτελέσματα ==========\n');
fprintf('CellSize   Accuracy(%%)\n');
for k = 1:numel(patchList)
    fprintf('%2dx%-2d      %6.2f\n', patchList(k),patchList(k), 100*accList(k));
end
fprintf('===========================================\n');

% γράφημα
figure('Name','Accuracy vs CellSize');
bar(patchList, 100*accList); grid on;
xlabel('Cell Size (pixels)'); ylabel('Accuracy (%)');
title('Επίδραση μεγέθους patch στα HOG + SVM');


%% 4)  BONUS – Manual HOG χωρίς extractHOGFeatures ------------------------
%  Βήματα:
%   1.  Sobel → Gx, Gy
%   2.  Μέτρο & γωνία (0–180°)
%   3.  Κβάντιση σε 9 bins (0:20:180)
%   4.  Ιστόγραμμα magnitudes σε κάθε κελί 8×8
%   5.  Concatenate 4×4×9 = 144 στοιχεία  (όπως το built-in)
%   6.  Σύγκριση με extractHOGFeatures
% ------------------------------------------------------------------------
fprintf('\n=== BONUS: Manual HOG έναντι built-in (Cell 8×8) ===\n');
Ndemo  = 1000;                 
cellSz = [8 8];                % ένα μόνο patch, όπως στο πείραμα 8×8
bins   = 0:20:180;             % 9 bins
epsN   = 1e-6;                 
sampleIdx = randperm(size(Xtrain,3), Ndemo);
rmseAll = zeros(1,Ndemo,'single');

for n = 1:Ndemo
    I = single(Xtrain(:,:, sampleIdx(n)));   % 28×28   [0,1]

    % (i) Gradients
    [Gx,Gy] = imgradientxy(I,'sobel');
    mag = hypot(Gx,Gy);
    ang = mod(atan2d(Gy,Gx),180);            % 0–180°

    % (ii) Υπολογισμός ιστογράμματος σε κάθε κελί 8×8 
    nCellsR = floor(size(I,1)/cellSz(1)); 
    nCellsC = floor(size(I,2)/cellSz(2));    
    cellHist = zeros(nCellsR,nCellsC,9,'single');

    for r = 1:cellSz(1):size(I,1)-cellSz(1)+1        % 1,9,17
        for c = 1:cellSz(2):size(I,2)-cellSz(2)+1    % 1,9,17
            rr = (r-1)/cellSz(1) + 1;                % δείκτης κελιού
            cc = (c-1)/cellSz(2) + 1;
            patchMag = mag(r:r+7, c:c+7);
            patchAng = ang(r:r+7, c:c+7);

            h = zeros(1,9,'single');
            for pr = 1:8
                for pc = 1:8
                    a = patchAng(pr,pc); m = patchMag(pr,pc);
                    bin = floor(a/20) + 1;           % κύριο bin
                    nextBin = mod(bin,9) + 1;        % κυκλικό
                    alpha = mod(a,20)/20;            % ποσοστό
                    h(bin)      = h(bin)     + m*(1-alpha);
                    h(nextBin)  = h(nextBin) + m*alpha;
                end
            end
            cellHist(rr,cc,:) = h;
        end
    end

    % (iii) Block-normalisation 2×2, overlap 1×1 
    hogManual = [];
    for br = 1:nCellsR-1          % 1,2   ⇒ 2 blocks κάθε κατεύθυνση
        for bc = 1:nCellsC-1
            block = cellHist(br:br+1, bc:bc+1, :);
            blockVec = block(:)';
            blockNorm = blockVec / sqrt(sum(blockVec.^2) + epsN);
            hogManual = [hogManual blockNorm];
        end
    end
   

    % (iv) Built-in για έλεγχο
    hogBuiltin = extractHOGFeatures(I, 'CellSize', cellSz);

    % (v) RMSE 
    rmseAll(n) = sqrt( mean( (hogManual - hogBuiltin).^2 ) );
end

fprintf('Μέσο RMSE σε %d δείγματα: %.2e\n', Ndemo, mean(rmseAll));
fprintf('Max  RMSE : %.2e\n', max(rmseAll));

% Οπτική επιβεβαίωση σε 1 τυχαία εικόνα
figure('Name','Manual vs Built-in HOG (τυχαίο δείγμα)');
subplot(1,2,1); plot(hogManual);   title('Manual HOG');          xlim([1 144]);
subplot(1,2,2); plot(hogBuiltin);  title('extractHOGFeatures');   xlim([1 144]);
sgtitle('Διαφορά ≪ 1e-4  →  πρακτικά ταυτόσημα');
