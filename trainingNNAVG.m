%% Team Members: Monkey See Monkey Do

function [modelParameters] = positionEstimatorTraining(training_data)
   
    %% Setup
    trial = training_data;                  % renames variables

    numNeurons = size(trial(1,1).spikes,1); % # neurons
    numTrials = size(trial,1);              % # trials per direction
    numDirections = size(trial,2);          % # directions experimented
    start = 320;                            % start of movement
    timestep = 20;
    maxIter = 18;                           % max number of movement steps

    %pre-allocates memory
    avgTraj = zeros(numDirections,2,maxIter);          %saves average trajectory
    %saves neural activty and direction (this matrix will train bagging model)
    classificationData = zeros( numTrials*numDirections , 2*numNeurons + 1 );    
    
    %% loops through trials & directions to fill in neural activity matrix
    for dir = 1 : numDirections
        
        HP = zeros(2,maxIter);      % stores Hand Position store
        count = zeros(1,maxIter);   % count frequency of trial time 
        
        for i = 1 : numTrials
            
            % takes firing from all neurons across time in 1 trial
            activity1 = trial(i,dir).spikes(:,1:start/2);          %activity from 0ms to 160ms
            firing1 = sum( activity1 , 2 )';                       %sums activity in first window
            activity2 = trial(i,dir).spikes(:,(start/2)+1:start);  %activity from 160ms to 320ms
            firing2 = sum( activity2 , 2 )';                       %sums activity in second window
            
            row = (dir-1)*numTrials + i ;    % determine row number
            %fills in row with direction and firing data
            classificationData(row,:) = [ dir ,  firing1 , firing2 ] ;
            
            pos = trial(i,dir).handPos(1:2,start:timestep:end); %get hand positions
            L = size(pos,2);                                    %get length of trial 
            
            %adds trial to hand pos array and increments time counter
            
            if L <= maxIter     %for shorter trials
                HP(:,1:L) = HP(:,1:L) + pos;
                count(1:L) = count(1:L) + ones(1,L);
            else                %for longer trials
                HP(:,1:maxIter) = HP(:,1:maxIter) + pos(:,1:maxIter);
                count(1:maxIter) = count(1:maxIter) + ones(1,maxIter);
            end

        end
        
        %caclulates average trajectory and removes nans
        for j = 1 : maxIter
            avgTraj(dir,:,j) = HP(:,j) ./ count(j) ;
            if isnan( avgTraj(dir,:,j) )
                avgTraj(dir,:,j) = avgTraj(dir,:,j-1);
            end
        end
    end
    %stores data in structure
    modelParameters.classificationData = classificationData;
    modelParameters.Avg = avgTraj;
    
end