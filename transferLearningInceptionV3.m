function [net] = transferLearningInceptionV3(net, Constants, ParametersTransferLearning)
    disp('Transfer learning');
    
    if(~isa(net,'DAGNetwork'))
        error('Variable net must be DAGNetwork');
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    
    if(isempty(ParametersTransferLearning) || isempty(ParametersTransferLearning.trainingOptions) || ...
            isempty(ParametersTransferLearning.initialLearnRate) || isempty(ParametersTransferLearning.miniBatchSize) || ...
            isempty(ParametersTransferLearning.maxEpochs) || isempty(ParametersTransferLearning.verbose) || ...
            isempty(ParametersTransferLearning.shuffle) || isempty(ParametersTransferLearning.validationPatience))
        error('Struct ParametersTansferLearning cannot be empty.'); 
    end

    path = strrep(Constants.path, 'KoNViD_1k_videos', '');
    
    images = imageDatastore(strcat(path,filesep,'SortedFrames'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    [trainFrames, valFrames] = splitEachLabel(images, 2/3, 'randomized');

    numClasses = numel(categories(trainFrames.Labels));

    lgraph = layerGraph(net);

    lgraph = removeLayers(lgraph, {'predictions_softmax','ClassificationLayer_predictions'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
        lgraph = addLayers(lgraph, newLayers);
        lgraph = connectLayers(lgraph,'predictions','fc');

    numIterationsPerEpoch = floor(numel(trainFrames.Labels)/ParametersTransferLearning.miniBatchSize);
    options = trainingOptions(ParametersTransferLearning.trainingOptions,'MiniBatchSize',...
        ParametersTransferLearning.miniBatchSize,'MaxEpochs',ParametersTransferLearning.maxEpochs,...
        'InitialLearnRate',ParametersTransferLearning.initialLearnRate,'Verbose',...
        ParametersTransferLearning.verbose,'Plots','training-progress',...
        'ValidationData',valFrames,'ValidationFrequency',numIterationsPerEpoch,...
        'ValidationPatience',ParametersTransferLearning.validationPatience,'Shuffle',...
        ParametersTransferLearning.shuffle,...
        'ExecutionEnvironment','gpu',...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,4));

    net = trainNetwork(trainFrames, lgraph, options);

end

