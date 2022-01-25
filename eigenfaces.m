function [accuracy, time] = eigenfaces(verbose)
    tic
    % loading data
    load classdata_train.mat
    load classdata_test.mat
    
%     assigning data
    trainer = grayfaces_train;
    test = grayfaces_test;
    
    % showing first 25 raw images
    if verbose
        figure;
        sgtitle("Raw Images")
        for i=1:25
            subplot(5, 5, i);
            imagesc(trainer(:, :, i*4))
            colormap('gray')
        end
    end
    
    % reshaping
    trainer = reshape(trainer, 64*64, 356)';
    test = reshape(test, 64*64, 356)';
    
    % mean center
    trainer = trainer - mean(trainer);
    test = test - mean(test);
    
    % covariance matrix
    covar = (1/4095) * (trainer' * trainer);
    
    % eigenthings
    [vectors, ~] = eigs(covar, 12);
    
    % showing eigenfaces 
    if verbose
        figure;
        for i = 1:12       
            subplot(3, 4, i);
            imagesc(reshape(vectors(:,i), [64 64]));
            colormap('gray')
        end
        sgtitle('Eigenfaces')
    end
        
    % projecting training data
    projectedTrain = vectors' * trainer';
    projectedTest = vectors' * test';
    
    % prediction + accuracy checking
    tracker = NaN(1,356);
    for test = 1:356
        distance=NaN(1, 356);
        for train = 1:356
            distance(train) = sqrt(sum((projectedTest(:, test)-projectedTrain(:, train)).^2));
        end
        [~, minIndex] = min(distance);
        tracker(test) = subject_test(minIndex) == subject_train(test);
    end
    accuracy = mean(tracker);
    time = toc;
end