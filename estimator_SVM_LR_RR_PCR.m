function [x, y, newParams] = positionEstimator_svm(test_data, modelParameters)

%% READ ME: 
% In the following file comment sections in / out to use or ignore them 
% For each sections instructions are in the comments 

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
        
        %Commented out for SVM due to NUM array, not CELL read-only array 
  %       princDir = cell2mat(princDir);      %convert cell value to correct form
%         princDir = str2double(princDir);
        
    else
        princDir = modelParameters.Direction;   %load direction  
    end  

%     %% Hand Position - Average Trajectory Method 
%     
%     timestep = 20;
%     maxIter = size(modelParameters.Avg,3); %number of iterations average trajectory models
%     iter = 1 + ((maxTime-start)/timestep); %calculates iteration step
%     
%     %starting hand position is always given
%     if maxTime == start       
%         x = test_data.startHandPos(1);
%         y = test_data.startHandPos(2);
%      
%     %decode position during movement to average trajectory  
%     elseif (maxIter >= iter) && (maxTime > start)
%         x = modelParameters.Avg(princDir,1,iter);
%         y = modelParameters.Avg(princDir,2,iter);
%         
%     %if max iterations exceeded use last average trajectory estimation
%     else
%         x = modelParameters.Avg(princDir,1,end);
%         y = modelParameters.Avg(princDir,2,end);
%     end
%     
%     % updates model parameters
      newParams = modelParameters;
      newParams.Direction = princDir;
    
    %% REGRESSION set parameters & coefficients from Training
    model_coeff_x=modelParameters.Coefficients_x(:, princDir); 
    model_coeff_y=modelParameters.Coefficients_y(:, princDir); 
  
    bins_before=5; 
    
    %% Uncomment to do LR without history using spikes 
%     test_spikes=test_data.spikes; 
%   
%     test_spikes=smoothdata(test_spikes,2, 'gaussian', 250);
% 
%     decoded_x= (model_coeff_x(2:end)' * test_spikes + model_coeff_x(1)); 
%     decoded_y= (model_coeff_y(2:end)' * test_spikes + model_coeff_y(1)); 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 250); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 250); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end); 
     %% Uncomment to do LR with history using spikes 
    
%     test_spikes=test_data.spikes; 
%     test_spikes=smoothdata(test_spikes,2, 'gaussian', 250);
%     test_spikes_with_history=get_history_matrix(test_spikes, 4);
%     test_spikes_with_history=smoothdata(test_spikes_with_history,2, 'gaussian', 250);
% 
%     decoded_x= model_coeff_x(2:end)' * test_spikes_with_history + model_coeff_x(1); 
%     decoded_y= model_coeff_y(2:end)' * test_spikes_with_history + model_coeff_y(1); 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 250); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 250); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end); 
     %% Uncomment to do LR without history using rates
    
%     numTrials=size(test_data, 1); 
%   
%     test_spikes=test_data.spikes; 
%     
%     % comment out to avoid smoothing, or change the std of the gaussian kernel
%     test_spikes=reshape(test_spikes, [1, size(test_spikes, 1), size(test_spikes, 2)]); 
%     avg_rates=find_rates2(test_spikes, 10);
%     avg_rates=smoothdata(avg_rates,2, 'gaussian', 300);
%     
%     decoded_x= model_coeff_x(2:end)' * avg_rates + model_coeff_x(1); 
%     decoded_y= model_coeff_y(2:end)' * avg_rates + model_coeff_y(1); 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 200); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 200); 
%   
%     x=decoded_x(end);
%     y=decoded_y(end); 

     %% Uncomment to do LR with history using rates
    
%     numTrials=size(test_data, 1); 
%   
%     test_spikes=test_data.spikes; 
%     
%     % comment out to avoid smoothing, or change the std of the gaussian kernel
%     
%     %test_spikes=smoothdata(test_spikes,2, 'gaussian', 300);
%     test_spikes=reshape(test_spikes, [1, size(test_spikes, 1), size(test_spikes, 2)]); 
%     
%     avg_rates=find_rates2(test_spikes, 10);
%     avg_rates_hist=get_history_matrix(avg_rates, 3); 
%     avg_rates_hist=smoothdata(avg_rates_hist,2, 'gaussian', 200);
%     
%     decoded_x= model_coeff_x(2:end)' * avg_rates_hist + model_coeff_x(1); 
%     decoded_y= model_coeff_y(2:end)' * avg_rates_hist + model_coeff_y(1); 
%     
%     decoded_x=smoothdata(decoded_x, 'gaussian', 200); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 200); 
%   
%     x=decoded_x(end); 
%     y=decoded_y(end); 

      %% Uncomment to do PCR using spikes
%     test_spikes=test_data.spikes; 
%     
%     % comment out to avoid smoothing, or change the std of the gaussian kernel
%     test_spikes=smoothdata(test_spikes,2, 'gaussian', 150);
%     
%     decoded_x= test_spikes' * model_coeff_x(2:end); 
%     decoded_y= test_spikes' * model_coeff_y(2:end); 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 200); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 200); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end); 
    
      %% Uncomment to do PCR using rates
%     test_spikes=test_data.spikes; 
%     
%     % comment out to avoid smoothing, or change the std of the gaussian kernel
%     test_spikes=smoothdata(test_spikes,2, 'gaussian', 200);
%     test_spikes=reshape(test_spikes, [1, size(test_spikes, 1), size(test_spikes, 2)]); 
%     avg_rates=find_rates2(test_spikes, 10);
%     avg_rates=smoothdata(avg_rates,2, 'gaussian', 300);
%         
%     decoded_x= avg_rates' * model_coeff_x(2:end); 
%     decoded_y= avg_rates' * model_coeff_y(2:end); 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 200); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 200); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end); 

    %% Uncomment this to do ridge regression on spikes with history 
    
%     test_spikes=test_data.spikes; 
%   
%     test_spikes_with_history=get_history_matrix(test_spikes, bins_before);
%     test_spikes_with_history=smoothdata(test_spikes_with_history,2, 'gaussian', 200);
% 
%     decoded_x= (model_coeff_x(2:end)' * test_spikes_with_history)'; 
%     decoded_y= (model_coeff_y(2:end)' * test_spikes_with_history)'; 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 100); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 100); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end);  

     %% Uncomment this to do ridge regression on rates with history 
             
%     numTrials=size(test_data, 1); 
%   
%     test_spikes=test_data.spikes; 
%     %test_spikes=smoothdata(test_spikes,2, 'gaussian', 180);
%     test_spikes=reshape(test_spikes, [1, size(test_spikes, 1), size(test_spikes, 2)]); 
%     avg_rates=find_rates2(test_spikes, 10);
%     
%     rates_hist=get_history_matrix(avg_rates, 5); 
%     rates_hist=smoothdata(rates_hist,2, 'gaussian', 200);
%     
%     decoded_x= (model_coeff_x(2:end)' * rates_hist)'; 
%     decoded_y= (model_coeff_y(2:end)' * rates_hist)'; 
%     
%     % comment out not to smooth the output trajectory 
%     decoded_x=smoothdata(decoded_x, 'gaussian', 200); 
%     decoded_y=smoothdata(decoded_y, 'gaussian', 200); 
%     
%     x=decoded_x(end); 
%     y=decoded_y(end); 
   %% FUNCTIONS
  function U_with_history=get_history_matrix(U, bins_before)
    numNeurons=size(U,1);
    reduced_time=size(U, 2) - (bins_before+1); 
    U_with_history=zeros(numNeurons*(bins_before+1),reduced_time); 
    for t=bins_before+1:reduced_time
        %U_with_history(:, t)=cat(1, U(:, t), U(:, t-1));
        U_hist_row=[U(:, t); U(:, t-1)]; 
        for i=2:bins_before
            U_hist_row=[U_hist_row; U(:, t-i)]; 
        end 
        U_with_history(:, t)=U_hist_row;
    end 
  end  

% Get input and output matrices 
function [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, data_struct, trainingSplit, validatingSplit, resolution, delay)
      
      bins_correcting=delay/resolution; % shifting matrices by how many bins 
      
      numNeurons = size(data_struct(1).spikes, 1); % # neurons
      numTrials = size(data_struct, 1);              % # trials per direction
      %numDirections = size(data_struct,2);          % # directions experimented

      % creates array to save the start and end of each trial
      duration = zeros(1,numTrials);          
      for i = 1 : numTrials
          duration(i) = length(data_struct(i).spikes);
      end
      % finds longest and shortest trials
      max_duration = max(duration); % 713 max duration of a trial
      min_duration = min(duration); % 593 min duration of a trial
      
      % create a 3D array of neural activity of neurons across trials (values are set to NAN)
      neural_activity = zeros(numTrials,numNeurons,min_duration - bins_correcting -320);
      
      % loops through trials and neurons to fill in neural activity matrix
      for i = 1 : numTrials
          for neuron = 1 : numNeurons
              % takes firing from 1 neuron in 1 trial
              activity = data_struct(i).spikes(neuron,320:min_duration-bins_correcting);
              % fills in activity to neural activity matrix
              neural_activity(i, neuron, 1:length(activity) ) = activity; 
          end 
      end 
      
      % Package Hand Data (OUTPUT FROM DECODER)

      % create array to save hand positions (note initially filled w/ NAN)
      pos = zeros(numTrials, 3, min_duration-bins_correcting -320); % note only x-y needed
      for i = 1 : numTrials
          for xyz = 1 : 3
              trajectory = data_struct(i).handPos(xyz,bins_correcting+320:min_duration);
              pos(i, xyz, 1:length(trajectory) ) = trajectory; 
          end 
      end 

      % Split dataset

      %indices of splitting
      trainEnd = floor(numTrials*trainingSplit);
      validateStart = floor(numTrials*(1-validatingSplit));

      % create training sets
      neural_activity_train = neural_activity(1:trainEnd, :, :);
      handpos_train=pos(1:trainEnd,:, :);
      
      % create validating sets
      neural_activity_validate=neural_activity(trainEnd+1:validateStart, :, :);
      handpos_validate=pos(trainEnd+1:validateStart,:, :);

      % create testing sets
      neural_activity_test=neural_activity(validateStart+1:numTrials, :, :);
      handpos_test=pos(validateStart+1:numTrials,:, :);

end

function avg_rates=find_rates2(neural_activity, dt)
    numTrials=size(neural_activity, 1);
    numNeurons=size(neural_activity, 2);
    limit=size(neural_activity, 3);
    
    % loop to ensure neural activity is in a format multiple of dt 
    for i=dt:dt:limit
        neural_reshaped=neural_activity(:, :, 1:i); 
    end
    
    % find firing rates 
    avg_rates=zeros(numTrials, numNeurons, size(neural_reshaped, 3)/dt); 
    for t=1:numTrials
        for n=1:numNeurons
            avg_rates(t, n, 1)=sum(neural_reshaped(t, n, 1:dt))/dt;
            for int=2:size(neural_reshaped, 3)/dt
                avg_rates(t, n, int)=sum(neural_reshaped(t, n, (dt-1)*int:dt*int))/dt;
            end
        end 
    end
    
    % average across trials 
    avg_rates=squeeze(mean(avg_rates, 1)); 
end 
end