function [x, y, newParams] = positionEstimator(test_data, modelParameters)

    %% Classification (Nearest Neighbor)
    
    testing_data = test_data.spikes;        %load data
    start = 320;                            %start of movement
    maxTime = size(testing_data,2);     %length of time of trial
    
    if maxTime == start         %at start of trajectory direction is estimated
        
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
    
    timestep = 20;
    maxIter = size(modelParameters.Avg,3); %number of iterations average trajectory models
    iter = 1 + ((maxTime-start)/timestep); %calculates iteration step
    N = 15;                                 %number of iterations for window
    window = N*timestep;                    %window length for regression

    xStart = test_data.startHandPos(1);     %starting x hand pos
    yStart = test_data.startHandPos(2);     %starting y hand pos
    
    if maxTime == start                         %first step is always given     
        x = xStart;
        y = yStart;
         
    elseif maxTime > start && iter <= maxIter        %decode position using avg and regression
        %find firing rate over time window
        activity = sum(testing_data(: , maxTime-window:maxTime) , 2);
        %use firing rate to predict change in x,y directions
        predX = predict( modelParameters.Position{princDir,1} , activity' );
        predY = predict( modelParameters.Position{princDir,2} , activity' );
        
        %use a 50/50 weightings of average traj & regressing pos change for new positions
        x = ((modelParameters.PosX + predX)*0.5 + (modelParameters.Avg(princDir,1,iter))*0.5);
        y = ((modelParameters.PosY + predY)*0.5 + (modelParameters.Avg(princDir,2,iter))*0.5);
        
    else    %decode hand pos using only regression as maxIter of average is exceeded
        
        %find firing rate over time window
        activity = sum(testing_data(: , maxTime-window:maxTime) , 2);
        
        %use firing rate to predict change in x,y directions
        predX = predict( modelParameters.Position{princDir,1} , activity' );
        predY = predict( modelParameters.Position{princDir,2} , activity' );
        
        %calculate new x,y position
        x = modelParameters.PosX + predX;
        y = modelParameters.PosY + predY;
        
        %SAFETY NET
        xDif = abs( xStart-x ); %calculate total x and y position change
        yDif = abs( yStart-y );
        % get average x and y position change from training data
        xTrainDif = abs (modelParameters.Avg(princDir,1,1) - modelParameters.Avg(princDir,1,end) );
        yTrainDif = abs (modelParameters.Avg(princDir,2,1) - modelParameters.Avg(princDir,2,end) );
        
        % if x,y position change is bigger than average fix position to end average trajectory
        if xDif+yDif > xTrainDif+yTrainDif
            x = modelParameters.Avg(princDir,1,end);
            y = modelParameters.Avg(princDir,2,end);
        end

    end
    
    % updates model parameters
    newParams = modelParameters;
    newParams.Direction = princDir;
    newParams.PosX = x;
    newParams.PosY = y;
    
end