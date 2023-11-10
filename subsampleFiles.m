function ads = subsampleFiles(ads, n)
% Get the total number of labels in each category
s = summary(ads.Labels);
cats = s.Emotion.Categories;
counts = s.Emotion.Counts;
cats = cats(counts > 0);
counts = counts(counts > 0);
% Check if the number of files is specified as an argument
if nargin > 1
    min_counts = n;
else
    min_counts = min(counts);
end
% Subsample the data, balancing the labels
rng(1);
idx_subsample = cell(1, numel(cats));
for c = 1:numel(cats)
    if counts(c) == min_counts
        idx_subsample{c} = find(contains(cellstr(ads.Labels.Emotion), cats(c)));
    else
        idx_all = find(contains(cellstr(ads.Labels.Emotion), cats(c)));
        idx_subsample{c} = idx_all(randperm(numel(idx_all), min_counts));
    end
end
% Get a subsampled Audio Datastore object based on indices
idx_subsample = vertcat(idx_subsample{:});
ads = subset(ads, idx_subsample);
end