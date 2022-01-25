function vectors = eigenvectors(pcs, smile_separated)
    if smile_separated == false
        load classdata_train.mat
        load classdata_test.mat
    else
        load classdata_no_smile.mat
        load classdata_smile.mat
    end
    
%     assigning data
    trainer = grayfaces_train;
    test = grayfaces_test;
    
    % reshaping
    trainer = reshape(trainer, 64*64, 356)';
    test = reshape(test, 64*64, 356)';
    
    % mean center
    trainer = trainer - mean(trainer);
    
    % covariance matrix
    covar = (1/4095) * (trainer' * trainer);
    
    % eigenthings
    [vectors, ~] = eigs(covar, pcs);
end