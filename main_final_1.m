clear all
clc
Name_ugc= table2cell(readtable('/home/hq/Downloads/RM3VQA/Dataset/video_name_final.csv'));
MOS_ugc= table2array(readtable('/home/hq/Downloads/RM3VQA/Dataset/videoname_mos_final.csv')); 
path = strcat(pwd,filesep,'Dataset'); % Dataset is the folder where the videos are stored.

% Parameters of the algorithm
%video_type='mp4';% please define the type of video that you have used
Constants.BaseArchitecture    = 'inceptionv3'; % 'inceptionv3' alternate is'inceptionresnetv2' 
Constants.PoolMethod          = 'max';         % 'avg', 'median', 'max', 'min' can be choosen

Constants.number_of_test_videos = 7;          % Please change this number as per the number of videos to be tested     

Constants.numberOfTrainVideos =5600;           % number of training videos
Constants.numberOfValidationVideos =700;      % number of validation videos 
Constants.numberOfVideos      = Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+Constants.number_of_test_videos;         % number of videos in the database 6300
Constants.path                = path;          % path to videos
Constants.useParallelToolbox  = true;          % true or false can be choose (to use or not to use Parallel Computing Toolbox)
Constants.useTransferLearning = true;          % to use transfer learning or not to use transfer learning (true or false)

c=parcluster;
c.NumWorkers=2;%initally it was 2
saveProfile(c);
delete(gcp('nocreate'));
parpool('local',2);

% Parameters for transfer learning
ParametersTransferLearning.trainingOptions    = 'sgdm';        % adam
ParametersTransferLearning.initialLearnRate   = 1e-4;          % 1e-5
ParametersTransferLearning.miniBatchSize      = 28;            % 32
ParametersTransferLearning.maxEpochs          = 10;            % 100 was alternate value, for test 1 and 40 are the options
ParametersTransferLearning.verbose            = false;
ParametersTransferLearning.shuffle            = 'every-epoch';
ParametersTransferLearning.validationPatience = Inf;
ParametersTransferLearning.N                  = 6;             % stops network
                                                               % training if the best
                                                               % classification accuracy on the validation
                                                               % data does not improve for N network
                                                               % validations in a row.

AllVideos = Name_ugc;

TrainVideos = Name_ugc(1:Constants.numberOfTrainVideos);%PermutedName(1:Constants.numberOfTrainVideos);
TrainMOS   = MOS_ugc(1:Constants.numberOfTrainVideos);%PermutedMOS(1:Constants.numberOfTrainVideos);
   
ValidationVideos = Name_ugc(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);
ValidationMOS    = MOS_ugc(Constants.numberOfTrainVideos+1:Constants.numberOfTrainVideos+Constants.numberOfValidationVideos);

TestMOS     = MOS_ugc((Constants.numberOfTrainVideos+Constants.numberOfValidationVideos+1):end);

% Loading pretrained CNN
if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
    load net_10epochs.mat
elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
    net = inceptionresnetv2;
else
    error('Unknown base architecture');
end

% Transfer learning
if(Constants.useTransferLearning)
        lgraph = layerGraph(net);
        %{
        %We should enable below line of code if we have to generate image from videos
       
        %%createTrainImages_ugc(TrainVideos, ValidationVideos, TrainMOS, ValidationMOS, Constants)
       
        if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
           
            path = strrep(Constants.path, 'Dataset', '');%Foldername where set of videos remains
           
            trainFrames = imageDatastore(strcat(path,filesep,'SortedFramesTrain'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            valFrames   = imageDatastore(strcat(path,filesep,'SortedFramesValidation'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

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
                'ExecutionEnvironment','gpu',...%gpu was replaced by CPU
                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,ParametersTransferLearning.N));

            %net = trainNetwork(trainFrames, lgraph, options);
           %}
            elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
           %{
            path = strrep(Constants.path, 'Dataset', '');%Path where dataset video remains
           
            trainFrames = imageDatastore(strcat(path,filesep,'SortedFramesTrain'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            valFrames   = imageDatastore(strcat(path,filesep,'SortedFramesValidation'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

            numClasses = numel(categories(trainFrames.Labels));
           %}
            lgraph = layerGraph(net);
%{
            lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});
            newLayers = [
                fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
                softmaxLayer('Name','softmax')
                classificationLayer('Name','classoutput')];
       
            lgraph = addLayers(lgraph, newLayers);
            lgraph = connectLayers(lgraph,'avg_pool','fc');

            numIterationsPerEpoch = floor(numel(train.Labels)/miniBatchSize);
            options = trainingOptions(ParametersTransferLearning.trainingOptions,'MiniBatchSize',...
                ParametersTransferLearning.miniBatchSize,'MaxEpochs',ParametersTransferLearning.maxEpochs,...
                'InitialLearnRate',ParametersTransferLearning.initialLearnRate,'Verbose',...
                ParametersTransferLearning.verbose,'Plots','training-progress',...
                'ValidationData',valFrames,'ValidationFrequency',numIterationsPerEpoch,'ValidationPatience',...
                ParametersTransferLearning.validationPatience,'Shuffle',ParametersTransferLearning.shuffle,...
                'ExecutionEnvironment','gpu',...%gpu is replaced by cpu
                'OutputFcn',@(info)stopIfAccuracyNotImproving(info,ParametersTransferLearning.N));

            %net = trainNetwork(trainFrames, lgraph, options);
     %}      
else
            error('Unknown base architecture');
        end
%end

%VideoLevelFeatures = getVideoLevelFeatures_ugc(AllVideos, net, Constants);
load VideoLevelFeatures.mat
TrainVideoLevelFeatures = VideoLevelFeatures(1:(Constants.numberOfTrainVideos+Constants.numberOfValidationVideos),:);
TestVideoLevelFeatures = getVideoLevelFeatures_ugc_test(AllVideos,net,Constants);
Mdl_cknn = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
%{
Mdl_cknn_mwm = fitcknn(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS],'NumNeighbors',3,...
   'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',true);
MdlLin = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'linear');
MdlGan = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');
MdlPoly_1 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 1);
MdlPoly_2 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
MdlPoly_3 = fitrsvm(TrainVideoLevelFeatures, [TrainMOS; ValidationMOS], 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3);

% Testing SVRs (linear, gaussian, 1st order polynomial, 2nd order
% polynomial, 3rd order polynomial)
YPredLin = predict(MdlLin, TestVideoLevelFeatures);
%}
YPredcknn= predict(Mdl_cknn, TestVideoLevelFeatures);
%{
YPredcknn_mwm= predict(Mdl_cknn_mwm, TestVideoLevelFeatures);
%YPredcknn_dist= predict(Mdl_cknn_dist, TestVideoLevelFeatures);

YPredGan = predict(MdlGan, TestVideoLevelFeatures);
YPredPoly_1 = predict(MdlPoly_1, TestVideoLevelFeatures);
YPredPoly_2 = predict(MdlPoly_2, TestVideoLevelFeatures);
YPredPoly_3 = predict(MdlPoly_3, TestVideoLevelFeatures);

PLCC.Linear = corr(YPredLin, TestMOS, 'Type', 'Pearson');
SROCC.Linear= corr(YPredLin, TestMOS, 'Type', 'Spearman');
KROCC.Linear= corr(YPredLin, TestMOS, 'Type', 'Kendall');

PLCC.Gaussian = corr(YPredGan, TestMOS, 'Type', 'Pearson');
SROCC.Gaussian= corr(YPredGan, TestMOS, 'Type', 'Spearman');
KROCC.Gaussian= corr(YPredGan, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_1 = corr(YPredPoly_1, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_1= corr(YPredPoly_1, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_1= corr(YPredPoly_1, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_2 = corr(YPredPoly_2, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_2= corr(YPredPoly_2, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_2= corr(YPredPoly_2, TestMOS, 'Type', 'Kendall');

PLCC.Polynomial_3 = corr(YPredPoly_3, TestMOS, 'Type', 'Pearson');
SROCC.Polynomial_3= corr(YPredPoly_3, TestMOS, 'Type', 'Spearman');
KROCC.Polynomial_3= corr(YPredPoly_3, TestMOS, 'Type', 'Kendall');
%}
PLCC = corr(YPredcknn, TestMOS, 'Type', 'Pearson');
%PLCC.cknn_mwm = corr(YPredcknn_mwm, TestMOS, 'Type', 'Pearson');
%PLCC.cknn_dist = corr(YPredcknn_dist, TestMOS, 'Type', 'Pearson');
SROCC= corr(YPredcknn, TestMOS, 'Type', 'Spearman');
%SROCC.cknn_mwm= corr(YPredcknn_mwm, TestMOS, 'Type', 'Spearman');
%SROCC.cknn_dist= corr(YPredcknn_dist, TestMOS, 'Type', 'Spearman');
KROCC= corr(YPredcknn, TestMOS, 'Type', 'Kendall');

%KROCC.cknn_mwm= corr(YPredcknn_mwm, TestMOS, 'Type', 'Kendall');
%KROCC.cknn_dist= corr(YPredcknn_dist, TestMOS, 'Type', 'Kendall');
RMSE=  sqrt(mean(YPredcknn-TestMOS)^2);

