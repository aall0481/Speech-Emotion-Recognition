function [trueLabelsCrossFold,predictedLabelsCrossFold] = HelperTrainAndValidateNetwork(varargin)
    % Copyright 2019-2023 The MathWorks, Inc.
    if nargin == 3
        ads = varargin{1};
        augads = varargin{2};
        extractor = varargin{3};
    elseif nargin == 2
        ads = varargin{1};
        augads = varargin{1};
        extractor = varargin{2};
    end
    speaker = categories(ads.Labels.Speaker);
    numFolds = numel(speaker);
    emptyEmotions = (ads.Labels.Emotion);
    emptyEmotions(:) = [];

    % Loop over each fold.
    trueLabelsCrossFold = {};
    predictedLabelsCrossFold = {};
    
    for i = 1:numFolds
        
        % 1. Divide the audio datastore into training and validation sets.
        % Convert the data to tall arrays.
        idxTrain = augads.Labels.Speaker~=speaker(i);
        augadsTrain = subset(augads,idxTrain);
        augadsTrain.Labels = augadsTrain.Labels.Emotion;
        idxValidation = ads.Labels.Speaker==speaker(i);
        adsValidation = subset(ads,idxValidation);
        adsValidation.Labels = adsValidation.Labels.Emotion;

        % 2. Extract features from the training set. Reorient the features
        % so that time is along rows to be compatible with
        % sequenceInputLayer.
        featuresTrain = extract(extractor,augadsTrain,UseParallel=canUseParallelPool);
        featuresTrain = cellfun(@(x)x',featuresTrain,UniformOutput=false);
        featuresValidation = extract(extractor,adsValidation,UseParallel=canUseParallelPool);
        featuresValidation = cellfun(@(x)x',featuresValidation,UniformOutput=false);

        % 3. Use the training set to determine the mean and standard
        % deviation of each feature. Normalize the training and validation
        % sets.
        allFeatures = cat(2,featuresTrain{:});
        M = mean(allFeatures,2,"omitnan");
        S = std(allFeatures,0,2,"omitnan");
        featuresTrain = cellfun(@(x)(x-M)./S,featuresTrain,UniformOutput=false);
        for ii = 1:numel(featuresTrain)
            idx = find(isnan(featuresTrain{ii}));
            if ~isempty(idx)
                featuresTrain{ii}(idx) = 0;
            end
        end
        featuresValidation = cellfun(@(x)(x-M)./S,featuresValidation,UniformOutput=false);
        for ii = 1:numel(featuresValidation)
            idx = find(isnan(featuresValidation{ii}));
            if ~isempty(idx)
                featuresValidation{ii}(idx) = 0;
            end
        end

        % 4. Buffer the sequences so that each sequence consists of twenty
        % feature vectors with overlaps of 10 feature vectors.
        featureVectorsPerSequence = 20;
        featureVectorOverlap = 10;
        [sequencesTrain,sequencePerFileTrain] = HelperFeatureVector2Sequence(featuresTrain,featureVectorsPerSequence,featureVectorOverlap);
        [sequencesValidation,sequencePerFileValidation] = HelperFeatureVector2Sequence(featuresValidation,featureVectorsPerSequence,featureVectorOverlap);

        % 5. Replicate the labels of the train and validation sets so that
        % they are in one-to-one correspondence with the sequences.
        labelsTrain = [emptyEmotions;augadsTrain.Labels];
        labelsTrain = labelsTrain(:);
        labelsTrain = repelem(labelsTrain,[sequencePerFileTrain{:}]);

        % 6. Define a BiLSTM network.
        dropoutProb1 = 0.3;
        numUnits = 200;
        dropoutProb2 = 0.6;
        layers = [ ...
            sequenceInputLayer(size(sequencesTrain{1},1))
            dropoutLayer(dropoutProb1)
            bilstmLayer(numUnits,OutputMode="last")
            dropoutLayer(dropoutProb2)
            fullyConnectedLayer(numel(categories(emptyEmotions)))
            softmaxLayer
            classificationLayer];

        % 7. Define training options.
        miniBatchSize = 512;
        initialLearnRate = 0.005;
        learnRateDropPeriod = 2;
        maxEpochs = 3;
        options = trainingOptions("adam", ...
            MiniBatchSize=miniBatchSize, ...
            InitialLearnRate=initialLearnRate, ...
            LearnRateDropPeriod=learnRateDropPeriod, ...
            LearnRateSchedule="piecewise", ...
            MaxEpochs=maxEpochs, ...
            Shuffle="every-epoch", ...
            Verbose=false);

        % 8. Train the network.
        net = trainNetwork(sequencesTrain,labelsTrain,layers,options);

        % 9. Evaluate the network. Call classify to get the predicted labels
        % for each sequence. Get the mode of the predicted labels of each
        % sequence to get the predicted labels of each file.
        predictedLabelsPerSequence = classify(net,sequencesValidation);
        trueLabels = categorical(adsValidation.Labels);
        predictedLabels = trueLabels;
        idx1 = 1;
        for ii = 1:numel(trueLabels)
            predictedLabels(ii,:) = mode(predictedLabelsPerSequence(idx1:idx1 + sequencePerFileValidation{ii} - 1,:),1);
            idx1 = idx1 + sequencePerFileValidation{ii};
        end
        trueLabelsCrossFold{i} = trueLabels; %#ok<AGROW>
        predictedLabelsCrossFold{i} = predictedLabels; %#ok<AGROW>
    end
end