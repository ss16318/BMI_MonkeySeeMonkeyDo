%% Team Members: Monkey See Monkey Do

function [modelParameters] = positionEstimatorTraining2(training_data)
   
    %% Setup
    
    trial = training_data;                  % renames variables

    numNeurons = size(trial(1,1).spikes,1); % # neurons
    numTrials = size(trial,1);              % # trials per direction
    numDirections = size(trial,2);          % # directions experimented
    start = 320;                            % start of movement
    timestep = 20;
    maxIter = 18;                           % max number of movement steps for average trajectory
    N = 15;                                 % # timesteps for time window
    window = N*timestep;                    % length of activity window for trajectory estimation
    Tree = 100;                             % # trees for grown for bagging regression

    %pre-allocates memory
    avgTraj = zeros(numDirections,2,maxIter);   %saves average trajectory
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
            
            for t = start : timestep : 560
                
                % finds activity from all neurons across time window in 1 trial
                activity = trial(i,dir).spikes(:,t-window:t);
                % determine firing rate
                firing = sum( activity , 2 )';
                
                X1 = trial(i,dir).handPos(1,t-window);  % xpos at start of time window
                X2 = trial(i,dir).handPos(1,t);         % xpos at end of time window
                Y1 = trial(i,dir).handPos(2,t-window);  % ypos at start of time window
                Y2 = trial(i,dir).handPos(2,t);         % ypos at end of time window

                xChange = (X2 - X1)/N; %change in xpos over time window (N timesteps)
                yChange = (Y2 - Y1)/N; %change in ypos over time window (N timesteps)
                
                %stores x,y position changes and firing activity (note time independence)
                posCh(i,:) = [ xChange , yChange , firing ];
                
            end
            
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
        
        %creates a trajectory model for each direction using bagging regression
        
        if dir == 1
            modelX1 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY1 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );
            
            modelParameters.Position{dir,1} = modelX1;
            modelParameters.Position{dir,2} = modelY1;
        
        elseif dir == 2
            modelX2 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY2 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );
            
            modelParameters.Position{dir,1} = modelX2;
            modelParameters.Position{dir,2} = modelY2;
        
        elseif dir == 3
            modelX3 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY3 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );
             
            modelParameters.Position{dir,1} = modelX3;
            modelParameters.Position{dir,2} = modelY3;
            
        elseif dir == 4
            modelX4 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY4 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );
            
            modelParameters.Position{dir,1} = modelX4;
            modelParameters.Position{dir,2} = modelY4;
    
        elseif dir == 5
            modelX5 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY5 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );   
            
            modelParameters.Position{dir,1} = modelX5;
            modelParameters.Position{dir,2} = modelY5;
        
        elseif dir == 6
            modelX6 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY6 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );  
            
            modelParameters.Position{dir,1} = modelX6;
            modelParameters.Position{dir,2} = modelY6;
        
        elseif dir == 7
            modelX7 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY7 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );
            
            modelParameters.Position{dir,1} = modelX7;
            modelParameters.Position{dir,2} = modelY7;
        
        elseif dir == 8
            modelX8 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,1) , 'Method' , 'Regression' );
            modelY8 = TreeBagger( Tree , posCh(:,3:end) , posCh(:,2) , 'Method' , 'Regression' );  
            
            modelParameters.Position{dir,1} = modelX8;
            modelParameters.Position{dir,2} = modelY8;

        end 
    end
    
    Classes = classificationData(:,1);
    Data = classificationData(:,2:end);
    
    %SVM
    classificationMdl2 = fitcecoc(Data,Classes,'Learners','svm');
    
    %stores data in structure
    modelParameters.Classification = classificationMdl2;
    modelParameters.Avg = avgTraj;

end