function [x, y, newParams] = est(test_data, modelParameters)

    %% Classification (Nearest Neighbor)
    
    testing_data = test_data.spikes;    %load data
    start = 320;                        %start of movement     
    maxTime = size(testing_data,2);     %length of time of trial
    
    if maxTime == start     %at start of trajectory direction is estimated
        
        %for each neuron sum firing from 0ms to 160ms
        firing1 = sum( testing_data(:,1:maxTime/2) , 2 );
        %for each neuron sum firing from 160ms to 320ms
        firing2 = sum( testing_data(:,(maxTime/2)+1:maxTime) , 2 );
         
        model = modelParameters.Classification;     %loads classification model params
        
        %combines firing activities in windows 1&2 and inputs this to classification model
        princDir = predict( model , [firing1' firing2'] );
        
    else
        princDir = modelParameters.Direction;   %load direction  
    end  

    %% Hand Position
    
    window = 100;   %window over which to spike count
    last = 570;     %last time window that LR will model
    
    %starting hand position is always given
    if maxTime == start       
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
     
    %decode position during movement with training data  
    elseif maxTime > start && maxTime < last
        %find spike count over time windows
        count1 = sum( testing_data(:,maxTime-window:maxTime) , 2 );
        count2 = sum( testing_data(:,maxTime-2*window:maxTime-window) , 2 );
        count3 = sum( testing_data(:,maxTime-3*window:maxTime-2*window) , 2 );
        
        input = [ count1' , count2' , count3' ];
        
        input = input * modelParameters.V(princDir).V;  %reproject data
        %choose relevant models
        model1 = modelParameters.Regression(princDir,1).model;
        model2 = modelParameters.Regression(princDir,2).model;
        %predict x,y position
        x = predict( model1 , input );
        y = predict( model2, input );
        
    else
        %use average last for longer trials
        x = modelParameters.lastPos(princDir,1);
        y = modelParameters.lastPos(princDir,2);
    
    end
    
    % updates model parameters
    newParams = modelParameters;
    newParams.Direction = princDir;
    
end