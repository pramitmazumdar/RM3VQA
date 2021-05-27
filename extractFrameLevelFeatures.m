function [FrameLevelFeatures] = extractFrameLevelFeatures(frame, net, Constants)
    % This function extract the frame-level features
    
    resolution = size(frame);
    
    if( length(resolution)~=3 || size(frame,3)~=3 )
        error('variable frame must be RGB image'); 
    end
    
    if(~isa(net,'DAGNetwork'))
        error('Variable net must be DAGNetwork');
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    
    %frame = imresize(frame, [338 338]);                            % resizing
    %frame = imcrop(frame, [170.5 30.5 298 298]);                    % cropping center patch
    FrameLevelFeatures = activations(net, frame, 'avg_pool');      % extracting activation values from CNN
    
    if(strcmp(Constants.BaseArchitecture, 'inceptionv3'))
        FrameLevelFeatures = reshape(FrameLevelFeatures, [1, 2048]);
    elseif(strcmp(Constants.BaseArchitecture, 'inceptionresnetv2'))
        FrameLevelFeatures = reshape(FrameLevelFeatures, [1, 1536]);
    else
        error('Unknown base architecture');
    end
end

