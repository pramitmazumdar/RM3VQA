function [] = createTrainImages_ugc(TrainVideos, ValidationVideos, TrainMOS, ValidationMOS, Constants)
    
    if(isempty(Constants) || isempty(Constants.BaseArchitecture) || ...
            isempty(Constants.PoolMethod) || isempty(Constants.numberOfVideos) || ...
            isempty(Constants.numberOfTrainVideos) || isempty(Constants.path) || ...
            isempty(Constants.useParallelToolbox) || isempty(Constants.useTransferLearning))
        error('Struct Constants cannot be empty.'); 
    end
    
    disp('Creating training images for transfer learning');
    numberOfTrainVideos = Constants.numberOfTrainVideos;
    numberOfValidationVideos = Constants.numberOfValidationVideos;
    
    path = Constants.path;
    path2 = strrep(Constants.path, 'ICME2021_UGC/avi_videos', 'SortedFramesTrain');
    
    
    if(exist('SortedFramesTrain') || exist('SortedFramesValidation'))
        delete(strcat('SortedFramesTrain', filesep, 'VeryGoodImages', filesep, '*.*'));
        delete(strcat('SortedFramesTrain', filesep, 'GoodImages', filesep, '*.*'));
        delete(strcat('SortedFramesTrain', filesep, 'MediocreImages', filesep, '*.*'));
        delete(strcat('SortedFramesTrain', filesep, 'PoorImages', filesep, '*.*'));
        delete(strcat('SortedFramesTrain', filesep, 'VeryPoorImages', filesep, '*.*'));
        
        delete(strcat('SortedFramesValidation', filesep, 'VeryGoodImages', filesep, '*.*'));
        delete(strcat('SortedFramesValidation', filesep, 'GoodImages', filesep, '*.*'));
        delete(strcat('SortedFramesValidation', filesep, 'MediocreImages', filesep, '*.*'));
        delete(strcat('SortedFramesValidation', filesep, 'PoorImages', filesep, '*.*'));
        delete(strcat('SortedFramesValidation', filesep, 'VeryPoorImages', filesep, '*.*'));
    else
        mkdir('SortedFramesTrain');
        mkdir(strcat('SortedFramesTrain', filesep, 'VeryGoodImages'));
        mkdir(strcat('SortedFramesTrain', filesep, 'GoodImages'));
        mkdir(strcat('SortedFramesTrain', filesep, 'MediocreImages'));
        mkdir(strcat('SortedFramesTrain', filesep, 'PoorImages'));
        mkdir(strcat('SortedFramesTrain', filesep, 'VeryPoorImages'));
        
        mkdir('SortedFramesValidation');
        mkdir(strcat('SortedFramesValidation', filesep, 'VeryGoodImages'));
        mkdir(strcat('SortedFramesValidation', filesep, 'GoodImages'));
        mkdir(strcat('SortedFramesValidation', filesep, 'MediocreImages'));
        mkdir(strcat('SortedFramesValidation', filesep, 'PoorImages'));
        mkdir(strcat('SortedFramesValidation', filesep, 'VeryPoorImages'));
    end
    
    i=1;
    
    for ind=1:numberOfTrainVideos
        if(mod(ind,10)==0)
            disp(ind); 
        end
             
        %pathnew=char(strcat(path,filesep, strrep(TrainVideos{ind,1},'''',''), '.mp4'));
        %pathnew1=char(strcat(path,filesep, convertStringToChars(TrainVideos{ind,1}), '.mp4'));
        %pathnew2=char(strcat(path,filesep, TrainVideos{ind,1}(2:end-1), '.mp4'));
        %pathnew=char(strcat(path,filesep, TrainVideos{ind,1}(2:end-1), '.webm'))
        
        v=VideoReader(char(strcat(path,filesep, TrainVideos{ind,1}(2:end-1), '.mp4')));
        %while hasFrame(v)
           % frame = readFrame(v);
        n= v.FrameRate*v.Duration;
        for iFrame=uint8(1:n/5)
            frame=read(v,5*iFrame);
            frame=imresize(frame, [340 340]);
            img=imcrop(frame, [20.5 20.5 298 298]);
            %img=salient_feature(frame);
            %frame=gbvs(frame);%GBVS model
            %img=cat(3,frame.master_map_resized, frame.master_map_resized, frame.master_map_resized);
            if(rand>=0.8)     
                        
                if(TrainMOS(ind)<=1.8)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryPoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>1.8 && TrainMOS(ind)<=2.6)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'PoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>2.6 && TrainMOS(ind)<=3.4)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'MediocreImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(TrainMOS(ind)>3.4 && TrainMOS(ind)<=4.2)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'GoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                else
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryGoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                end
          
                i=i+1;
                %clear('saveIm');
                %clear('img');
            end
            
           end
           %clear('frame');
        %clear('v');
        
    end  
    
path2 = strrep(Constants.path, 'ICME2021_UGC/avi_videos', 'SortedFramesValidation'); 
    i=1;
    for ind=1:numberOfValidationVideos
        if(mod(ind,10)==0)
            disp(ind); 
        end
        v=VideoReader( char(strcat(path, filesep, ValidationVideos{ind,1}(2:end-1), '.mp4')) );
        n=v.FrameRate*v.Duration;
    
        %while hasFrame(v)
         %   frame = readFrame(v);
         for iFrame=uint8(1:n/5)
            frame=read(v,5*iFrame); %frame=gbvs(frame);
            frame=imresize(frame, [360,640]);
            frame=imcrop(frame, [170.5 30.5 298 298]);
           %frame=gbvs(frame);%GBVS model
            img=salient_feature(frame);
            %img=cat(3,frame.master_map_resized, frame.master_map_resized, frame.master_map_resized);
            if(rand>=0.8)     
                        
                if(ValidationMOS(ind)<=1.8)*-9
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryPoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(ValidationMOS(ind)>1.8 && ValidationMOS(ind)<=2.6)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'PoorImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(ValidationMOS(ind)>2.6 && ValidationMOS(ind)<=3.4)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'MediocreImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                elseif(ValidationMOS(ind)>3.4 && ValidationMOS(ind)<=4.2)
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'GoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                else
                    %img = imresize(frame,[360 640]);
                    %img = imcrop(img, [170.5 30.5 298 298]);
                    saveIm = strcat(path2, filesep, 'VeryGoodImages',filesep,int2str(i),'.jpg');
                
                    imwrite(img, saveIm);
                end
            
                i=i+1;
            
            end
         end 
    end

end


