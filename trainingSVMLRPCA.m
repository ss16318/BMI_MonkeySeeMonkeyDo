%% Team Members: Monkey See Monkey Do

function [modelParameters] = positionEstimatorTraining(training_data)

%% Setup
trial = training_data;                  % renames variables

numNeurons = size(trial(1,1).spikes,1); % # neurons
numTrials = size(trial,1);              % # trials per direction
numDirections = size(trial,2);          % # directions experimented
start = 320;                            % start of movement
maxTime = 600;

%pre-allocates memory
%saves neural activty and direction (this matrix will train bagging model)
classificationData = zeros( numTrials*numDirections , 2*numNeurons + 1 );

window = 100;                           %window used for regression

measurements = (3*numNeurons);          %number of features
%neural data for spiking
regressData = zeros(numDirections,numTrials*(maxTime-start)/window,measurements);

%% loops through trials & directions to fill in neural activity matrix
for dir = 1 : numDirections
    
    j=1;    %indexes row of neural data matrix
    endPos = [ 0 ; 0 ]; %end position counter
    
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
        
        for time = start : 20 :maxTime  %samples every 20ms
            
            if trialLength >= time      %trial time must be at least 600ms
                %spike count measurements
                count1 = sum( trial(i,dir).spikes(:,time-window:time) , 2 );
                count2 = sum( trial(i,dir).spikes(:,time-2*window:time-window) , 2 );
                count3 = sum( trial(i,dir).spikes(:,time-3*window:time-2*window) , 2 );
                
                newData = [ count1' , count2' , count3' ];
                
                regressData(dir,j,:) = newData; %add meausurements to neural data matrix 
                
                pos = trial(i,dir).handPos(1:2,time);   %takes position data
                
                posData(j,:) = pos';
                
                j = j+1;    %increment row index
                 
            end     
        end
        
        endPos = endPos + trial(i,dir).handPos(1:2,end);    %add end position data
    end
    
    avgEndPos(dir,:) = endPos'/numTrials;   %calculates average end position at for each direction
    
    RD = squeeze ( regressData(dir,:,:) - mean(regressData(dir,:,:)) ); %mean-centre data
    
    [U S V] = svd(RD , 'econ'); %perform SVD
    
    %This part finds explained variance
%     L = S.^2 / (numTrials-1);
%     
%     Ltotal = sum(sum(L));
%     CumSum = 0;
%     k = 0;
%     
%     while k <= 10
%         k = k+1;
%         CumSum = (CumSum + L(k,k));
%         
%         Explained = CumSum/Ltotal;
%     end
    
    PCAdata = squeeze(regressData(dir,:,:)) * V(:,1:10) ;   %reprojects data into neural space
    
    modelParameters.V(dir).V = V(:,1:10);                   %saves reprojection transform (eigenvectors used)
    
    %creates models for each direction
    if dir == 1
        model1X = fitlm(PCAdata,posData(:,1));
        model1Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(1,1).model = model1X;
        modelParameters.Regression(1,2).model = model1Y;
        
        
    elseif dir == 2
        model2X = fitlm(PCAdata,posData(:,1));
        model2Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(2,1).model = model2X;
        modelParameters.Regression(2,2).model = model2Y;
    elseif dir == 3
        model3X = fitlm(PCAdata,posData(:,1));
        model3Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(3,1).model = model3X;
        modelParameters.Regression(3,2).model = model3Y;
    elseif dir == 4
        model4X = fitlm(PCAdata,posData(:,1));
        model4Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(4,1).model = model4X;
        modelParameters.Regression(4,2).model = model4Y;
    elseif dir == 5
        model5X = fitlm(PCAdata,posData(:,1));
        model5Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(5,1).model = model5X;
        modelParameters.Regression(5,2).model = model5Y;
    elseif dir == 6
        model6X = fitlm(PCAdata,posData(:,1));
        model6Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(6,1).model = model6X;
        modelParameters.Regression(6,2).model = model6Y;
    elseif dir == 7
        model7X = fitlm(PCAdata,posData(:,1));
        model7Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(7,1).model = model7X;
        modelParameters.Regression(7,2).model = model7Y;
    else
        model8X = fitlm(PCAdata,posData(:,1));
        model8Y = fitlm(PCAdata,posData(:,2));
        
        modelParameters.Regression(8,1).model = model8X;
        modelParameters.Regression(8,2).model = model8Y;
    end
    
end

Classes = classificationData(:,1);
Data = classificationData(:,2:end);

%SVM
classificationMdl2 = fitcecoc(Data,Classes,'Learners','svm');
%stores data in structure
modelParameters.Classification = classificationMdl2;

modelParameters.lastPos = avgEndPos;
end