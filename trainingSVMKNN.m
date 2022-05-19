%% Team Members: Monkey See Monkey Do

function [modelParameters] = positionEstimatorTraining(training_data)
   
    %% Setup
    trial = training_data;                  % renames variables

    numNeurons = size(trial(1,1).spikes,1); % # neurons
    numTrials = size(trial,1);              % # trials per direction
    numDirections = size(trial,2);          % # directions experimented
    start = 320;                            % start of movement
    timestep = 20;
    maxTime = 660;

    %pre-allocates memory
    %saves neural activty and direction (this matrix will train bagging model)
    classificationData = zeros( numTrials*numDirections , 2*numNeurons + 1 );
    
    window = 100;                           %window used for regression
    
    %% loops through trials & directions to fill in neural activity matrix
    for dir = 1 : numDirections
        
        for i = 1 : numTrials
            
            % takes firing from all neurons across time in 1 trial
            activity1 = trial(i,dir).spikes(:,1:start/2);          %activity from 0ms to 160ms
            firing1 = sum( activity1 , 2 )';                       %sums activity in first window
            activity2 = trial(i,dir).spikes(:,(start/2)+1:start);  %activity from 160ms to 320ms
            firing2 = sum( activity2 , 2 )';                       %sums activity in second window
            
            row = (dir-1)*numTrials + i ;    % determine row number
            %fills in row with direction and firing data
            classificationData(row,:) = [ dir ,  firing1 , firing2 ] ;
            
            trialLength = size( trial(i,dir).handPos , 2);  %gets length of trial
            
             for time = start : timestep : maxTime
                
                if trialLength >= time     %if trial is running
                    
                    %find spike count activity
                    count1 = sum( trial(i,dir).spikes(:,time-window:time) , 2 );
                    count2 = sum( trial(i,dir).spikes(:,time-2*window:time-window) , 2 );
                    count3 = sum( trial(i,dir).spikes(:,time-3*window:time-2*window) , 2 );
                    
                    pos = trial(i,dir).handPos(1:2,time); %get hand position
                    %add to position data matrix
                    posData(dir,i,time,:) = [ pos' , count1' , count2' , count3' ];
                    
                else %if trial has stopped take last spike windows and last hand position
                    count1 = sum( trial(i,dir).spikes(:,trialLength-window:trialLength) , 2 );
                    count2 = sum( trial(i,dir).spikes(:,trialLength-2*window:trialLength-window) , 2 );
                    count3 = sum( trial(i,dir).spikes(:,trialLength-3*window:trialLength-2*window) , 2 );
                    
                    pos = trial(i,dir).handPos(1:2,trialLength); %get hand positions
                    posData(dir,i,time,:) = [ pos' , count1' , count2' , count3' ];
                end
             end 
        end
    end
    
    Classes = classificationData(:,1);
    Data = classificationData(:,2:end);
    
    %SVM
    classificationMdl2 = fitcecoc(Data,Classes,'Learners','svm');
    
    %stores data in structure
    modelParameters.Classification = classificationMdl2;
    modelParameters.posData = posData;
end