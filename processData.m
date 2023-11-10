function [afe, sequencesTrain, labelsTrain, emptyEmotions] = processData(datasetName, ads, fs, numAugmentations)
% Build the augmenter
augmenter = audioDataAugmenter(NumAugmentations=numAugmentations, ...
    TimeStretchProbability=0, ...
    VolumeControlProbability=0, ...
    ...
    PitchShiftProbability=0.5, ...
    ...
    TimeShiftProbability=1, ...
    TimeShiftRange=[-0.3,0.3], ...
    ...
    AddNoiseProbability=1, ...
    SNRRange=[-20,40]);
% Create a directory to store the augmented data
currentDir = pwd;
writeDirectory = fullfile(currentDir, sprintf("augmentedData%s", datasetName));
mkdir(writeDirectory)
% Augment and save the data
reset(ads)
numPartitions = 18;
tic
parfor ii = 1:numPartitions
    adsPart = partition(ads,numPartitions,ii);
    while hasdata(adsPart)
        [x,adsInfo] = read(adsPart);
        data = augment(augmenter,x,fs);

        [~,fn] = fileparts(adsInfo.FileName);
        for i = 1:size(data,1)
            augmentedAudio = data.Audio{i};
            augmentedAudio = augmentedAudio/max(abs(augmentedAudio),[],"all");
            augNum = num2str(i);
            if numel(augNum)==1
                iString = ['0',augNum];
            else
                iString = augNum;
            end
            audiowrite(fullfile(writeDirectory,sprintf('%s_aug%s.wav',fn,iString)),augmentedAudio,fs);
        end
    end
end
disp("Augmentation complete in " + round(toc/60,2) + " minutes.")
% Read the augmented files and re-label
adsAug = audioDatastore(writeDirectory);
adsAug.Labels = repelem(ads.Labels,augmenter.NumAugmentations,1);
% Build the feature extractor
win = hamming(round(0.03*fs),"periodic");
overlapLength = 0;
afe = audioFeatureExtractor( ...
    Window=win, ...
    OverlapLength=overlapLength, ...
    SampleRate=fs, ...
    ...
    gtcc=true, ...
    gtccDelta=true, ...
    mfccDelta=true, ...
    ...
    SpectralDescriptorInput="melSpectrum", ...
    spectralCrest=true);
% Extract features from the augmented files
adsTrain = adsAug;
featuresTrain = extract(afe, adsTrain, UseParallel=true);
goodTrain = cellfun(@ndims, featuresTrain) < 3;
featuresTrain = featuresTrain(goodTrain);
labels = adsTrain.Labels;
adsTrain.Files = adsTrain.Files(goodTrain);
adsTrain.Labels = labels(goodTrain, :);
featuresTrain = cellfun(@(x)x',featuresTrain,UniformOutput=false);
% Z-score the features
allFeatures = cat(2,featuresTrain{:});
M = mean(allFeatures,2,"omitnan");
S = std(allFeatures,0,2,"omitnan");
featuresTrain = cellfun(@(x)(x-M)./S,featuresTrain,UniformOutput=false);
% Transform the features to sequences
featureVectorsPerSequence = 20;
featureVectorOverlap = 10;
[sequencesTrain,sequencePerFileTrain] = HelperFeatureVector2Sequence(featuresTrain,featureVectorsPerSequence,featureVectorOverlap);
% Get the training labels
labelsTrain = repelem(adsTrain.Labels.Emotion,[sequencePerFileTrain{:}]);
emptyEmotions = ads.Labels.Emotion;
emptyEmotions(:) = [];
end