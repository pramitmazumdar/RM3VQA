function [VideoLevelFeatures] = extractVideoLevelFeatures(video, net, Constants)
    % This function extracts the video-level features    
    
    if(~isa(net,'DAGNetwork'))
        error('Variable net must be DAGNetwork');
    end
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    
    % Extracting all frame-level features of the video sequence
     %n=video.FrameRate*video.Duration
     n=110;
     frame_Space=2;
     n=uint8((n/frame_Space)-0.5);
     %n=2; %v.numFames
        %while hasFrame(v)
           % frame = readFrame(v);
    k=1;
    %FrameLevelFeatures = zeros( ceil(video.FrameRate*video.Duration), 2048);
    FrameLevelFeatures = zeros( n, 2048);
    
    %while hasFrame(video)
        %frame = readFrame(video);
         for iFrame=1:n
            frame=read(video,(frame_Space*iFrame-frame_Space+1));
            %frame=imresize(frame,[360 640]);
            frame=imresize(frame,[340 340]);
            %frame=imcrop(frame, [170.5 30.5 298 298]);
            frame=imcrop(frame, [20.5 20.5 298 298]);
            %frame=salient_feature(frame);
             % frame=imresize(frame,[299 299]);
             %frame=gbvs(frame);%gbvs
             %%frame=simpsal(frame);%simple saliency
             %frame=mat2gray(imresize( frame, [360 640] ));
             %%frame=mat2gray(imresize( frame, [299 299] ));
             %frame=cat(3,frame.master_map_resized, frame.master_map_resized, frame.master_map_resized);
             %%frame=cat(3,frame,frame,frame);
        FrameLevelFeatures(k,:) = extractFrameLevelFeatures(frame, net, Constants);
        k=k+1;
        end
    FrameLevelFeatures = FrameLevelFeatures(1:(k-1), :);
    
    % Choosing pooling method and compiling video-level feature vector
    % of the video sequence
    if(strcmp(Constants.PoolMethod, 'max'))
        VideoLevelFeatures = max(FrameLevelFeatures,[],1); % max pooling
    elseif(strcmp(Constants.PoolMethod, 'min'))
        VideoLevelFeatures = min(FrameLevelFeatures,[],1); % min pooling
    elseif(strcmp(Constants.PoolMethod, 'median'))
        VideoLevelFeatures = median(FrameLevelFeatures,1); % median pooling
    elseif(strcmp(Constants.PoolMethod, 'avg'))
        VideoLevelFeatures = mean(FrameLevelFeatures,1);   % average pooling
    else
        error('Unknown base architecture');
    end
end

