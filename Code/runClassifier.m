function outputLabel = runClassifier(FeatureList)

    % The classifier has been trained with 43 features : time and 42
    % gyroscope readings. Hence the input FeatureList is Nx43 matrix
    
    dim = size(FeatureList); % acquiring the dimentions of the input matrix
    outputLabel = zeros(dim(1),1); % Setting the default null o/p
    
    % Since our algo is trained using 43 features per sample,
    % we need to check if each of the test data in the i/p has 43 features.
    if dim(2) ~= 43
        disp('43 features expected per sample');
        return;
    end
    
    %% Handling NaN in features
    
    feat = FeatureList(:,2:end);
    
    [m,n] = size(feat);

    for i = 1:m
        if sum(isnan(feat(i,:))) ~= 0
            for j=1:n
                if (isnan(feat(i,j)) == 1)
                    feat(i,j) = feat(i-1,j);
                end
            end
        end    
    end
    
    %% Running KNN on the test data set
    
    load('Trained_KNN.mat');
    
    op = model.predictFcn(FeatureList);
    op = op+1;
    
    %% Running HMM on the output of KNN
    
    load('Trained_HMM.mat');
    PSTATES = hmmdecode(op',ESTTR,ESTEMIT);
    
    res = [];
    for i=1:size(PSTATES,2)
        [~,in] = max(PSTATES(:,i));
        res = [ res in ];
    end
    
    outputLabel = res; 
end