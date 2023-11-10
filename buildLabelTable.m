function labelTable = buildLabelTable(ads, emotionIdx, speakerIdx, emotionMapping)
% Check whether the code is one or two coordinates (in file name) and get
% from indices
if numel(emotionIdx) == 1
    emotionCodes = cellfun(@(x)x(end-emotionIdx),ads.Files,UniformOutput=false);
else
    emotionCodes = cellfun(@(x)x(end-emotionIdx(1):end-emotionIdx(2)),ads.Files,UniformOutput=false);
end
if numel(emotionIdx) == 1
    speakerCodes = cellfun(@(x)x(end-speakerIdx),ads.Files,UniformOutput=false);
else
    speakerCodes = cellfun(@(x)x(end-speakerIdx(1):end-speakerIdx(2)),ads.Files,UniformOutput=false);
end
% Map the codes to emotion names
emotions = replace(emotionCodes, emotionMapping{1}, emotionMapping{2});
% Build the label table
labelTable = cell2table([speakerCodes,emotions],VariableNames=["Speaker","Emotion"]);
labelTable.Emotion = categorical(labelTable.Emotion);
labelTable.Speaker = categorical(labelTable.Speaker);
end