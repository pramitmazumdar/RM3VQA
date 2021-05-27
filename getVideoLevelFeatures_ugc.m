function [VideoLevelFeatures] = getVideoLevelFeatures_ugc(AllVideos, net, Constants)
     % This function extracts video-level feature vectors from a set of video
    % sequences
    
    
    if(~isa(net,'DAGNetwork'))
        error('Variable net must be DAGNetwork');
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
           isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
           isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    

    numberOfVideos = Constants.numberOfVideos;
    %numberOfVideos=7100
    
    % Choosing CNN base architecture
   
    if(strcmp(Constants.BaseArchitecture,'inceptionv3'))
        VideoLevelFeatures = zeros(numberOfVideos, 2048);
    elseif(strcmp(Constants.BaseArchitecture,'inceptionresnetv2'))
        VideoLevelFeatures = zeros(numberOfVideos, 1536);
    else
        error('Unknown base architecture');
    end
    
    if(Constants.useParallelToolbox)
            for i=1:numberOfVideos
            if(mod(i,10)==0)
                disp(i);
            end
            path_test=char(strcat(Constants.path, filesep, AllVideos{i,1}(2:end-1), '.mp4'))
            video=VideoReader( char(strcat(Constants.path, filesep, AllVideos{i,1}(2:end-1), '.mp4')) );%
            VideoLevelFeatures(i,:) = extractVideoLevelFeatures(video, net, Constants);
            end
    else
        for i=1:numberOfVideos
            if(mod(i,10)==0)
                disp(i);
            end
            video=VideoReader( char(strcat(Constants.path, filesep, AllVideos{i,1}(2:end-1), '.mp4')) );
            VideoLevelFeatures(i,:) = extractVideoLevelFeatures(video, net, Constants);
        end
    end
    
%}
end

