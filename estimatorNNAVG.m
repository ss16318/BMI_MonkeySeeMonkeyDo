function [x, y, newParams] = positionEstimator(test_data, modelParameters)

    %% Classification (Nearest Neighbor)
    
    testing_data = test_data.spikes;    %load data
    start = 320;                        %start of movement     
    maxTime = size(testing_data,2);     %length of time of trial
    
    if maxTime == start     %at start of trajectory direction is estimated
        
        %for each neuron sum firing from 0ms to 160ms
        firing1 = sum( testing_data(:,1:maxTime/2) , 2 );
        %for each neuron sum firing from 160ms to 320ms
        firing2 = sum( testing_data(:,(maxTime/2)+1:maxTime) , 2 );
        %load classification data 
        classificationData = modelParameters.classificationData;
        
        k = 1;  %determine number of nearest neighbours
        %find indices of nearest trials 
        Idx = knnsearch( classificationData(:,2:end) , [ firing1' , firing2' ] , 'K' , k );
        %pick most common class
        princDir = round(mode( classificationData(Idx,1) ));
        
    else
        princDir = modelParameters.Direction;   %load direction  
    end  

    %% Hand Position
    
    timestep = 20;
    maxIter = size(modelParameters.Avg,3); %number of iterations average trajectory models
    iter = 1 + ((maxTime-start)/timestep); %calculates iteration step
    
    %starting hand position is always given
    if maxTime == start       
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
     
    %decode position during movement to average trajectory  
    elseif (maxIter >= iter) && (maxTime > start)
        x = modelParameters.Avg(princDir,1,iter);
        y = modelParameters.Avg(princDir,2,iter);
        
    %if max iterations exceeded use last average trajectory estimation
    else
        x = modelParameters.Avg(princDir,1,end);
        y = modelParameters.Avg(princDir,2,end);
    end
    
    % updates model parameters
    newParams = modelParameters;
    newParams.Direction = princDir;
    
end