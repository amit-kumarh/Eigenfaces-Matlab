function accuracy = indAccuracy(vectors, smile_separated)
    if smile_separated == false
        load classdata_train.mat
        load classdata_test.mat
    else
        load classdata_no_smile.mat
        load classdata_smile.mat
    end
    trainer = grayfaces_train;
    test = grayfaces_test;
    
    % reshaping
    trainer = reshape(trainer, 64*64, 356)';
    test = reshape(test, 64*64, 356)';
    
    % mean center
    trainer = trainer - mean(trainer);
    test = test - mean(test);

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
end