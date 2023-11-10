function [ads_train, ads_test] = splitTrainTest(ads, propTrain)
% Get the total number of labels in each category
s = summary(ads.Labels);
cats = s.Emotion.Categories;
counts = s.Emotion.Counts;
% Distribute the data to train and test sets, balancing the labels
rng(1);
idx_train = cell(1, numel(cats));
idx_test = cell(1, numel(cats));
for c = 1:numel(cats)
    N = counts(c);
    idx_all = find(contains(cellstr(ads.Labels.Emotion), cats(c)));
    train_idx = randperm(numel(idx_all), floor(N*propTrain));
    idx_train{c} = idx_all(train_idx);
    idx_test{c} = idx_all(setdiff(1:numel(idx_all), train_idx));
end
% Get Audio Datastore objects for the train and test data based on file indices
idx_train = vertcat(idx_train{:});
idx_test = vertcat(idx_test{:});
ads_train = subset(ads, idx_train);
ads_test = subset(ads, idx_test);
end