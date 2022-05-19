%% Team Members: Monkey See Monkey Do

%% READ ME: 
% In the following file comment sections in / out to use or ignore them 
% Also changes to smoothing & delay functions can be made 
% For each sections instructions are in the comments 

%%
function [modelParameters] = positionEstimatorTraining_svm(training_data)
   
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
        
        HP = zeros(2,maxIter);      % stores Hand Position
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
            
%             %creating a table with the classificationData
%             Tbl=table(classificationData)
            
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
    
    Classes=classificationData(:,1);
    Data=classificationData(:,2:end);
    
    %Bagging (using 50 trees)
    %classificationMdl = TreeBagger( 50 , Data , Classes , 'Method' , 'Classification' );
    
    %SVM
    classificationMdl2=fitcecoc(Data,Classes,'Learners','svm');
    
    % Naive-Bayesian (below): does not work because it says that the variance is zero?
    % classificationMdl3=fitcnb(Data,Classes);
    %Alternatively, naive bayesian can be done by changing the svm -> naivebayes
    
    %Boosting
    %cellClasses=num2cell(int2str(Classes));
    %classificationMdl4 = fitensemble(Data,cellClasses,'AdaBoostM2',150,'Tree');
    
    %stores data in structure
    modelParameters.Classification = classificationMdl2;
    modelParameters.Avg = avgTraj;
    
    %% REGRESSION 
    
    % initialise parameters 
    bins_before=5; 
    numNeurons_hist= (bins_before+1)*size(training_data(1,1).spikes, 1);
  
    % Package initial neural data

%     %pre-allocates memory
%     training = zeros( numDirections , numTrials , numNeurons);
% 
%     % loops through trials & directions to fill in neural activity matrix
%     for dir = 1 : numDirections
%         for i = 1 : numTrials
%             % takes firing from all neurons across time in 1 trial
%             activity = training_data(i,dir).spikes(:,1:duration);
%             % sums all the spikes from neurons & saves in training_data file
%             training(dir , i , : ) = sum( activity , 2 )';
%         end 
%     end
        
    for dir=1:numDirections
%% ------ Uncomment this part is for LR without history using spikes as input field 
      
%       % change last input argument to add a neural delay in seconds 
%       [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train,handpos_validate]=get_io_matrices2(dir, training_data, 1, 0, 0.001, 0.01);
%       neural_activity_train = remove_nans(neural_activity_train); 
%       handpos_train = remove_nans(handpos_train); 
%       
%       % average across trials 
%       avg_train=squeeze(mean(neural_activity_train, 1));
%       avg_pos_train=squeeze(mean(handpos_train, 1));
%       
%       avg_train(isnan(avg_train))=0;
%       avg_pos_train(isnan(avg_pos_train))=0;
%      
%       % train linear regression model and get coefficients 
% 
%       avg_train = avg_train';
%       pos_train = avg_pos_train';
%       
%       % fit model 
%       mdl_x = fitlm(avg_train, pos_train(:, 1)); 
%       mdl_y = fitlm(avg_train, pos_train(:, 2)); 
%       
%       % get model coefficients 
%       model_x_coeffs(:, dir)=mdl_x.Coefficients.Estimate; 
%       model_y_coeffs(:, dir)=mdl_y.Coefficients.Estimate; 
%% ------ Uncomment this part is for LR with history using spikes as input field 
      
%       % change last input argument to add a neural delay in seconds 
%       [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train,handpos_validate]=get_io_matrices2(dir, training_data, 1, 0, 0.001, 0.01);
%       neural_activity_train = remove_nans(neural_activity_train); 
%       handpos_train = remove_nans(handpos_train); 
%       
%       % average across trials 
%       avg_train=squeeze(mean(neural_activity_train, 1));
%       avg_pos_train=squeeze(mean(handpos_train, 1));
%       
%       avg_train(isnan(avg_train))=0;
%       avg_pos_train(isnan(avg_pos_train))=0;
%      
%       % get matrices with/for history
%       avg_train_with_history=get_history_matrix(avg_train, 4);
%       pos_train_for_history = avg_pos_train(:, 4+2:end); 
%       
%       % train linear regression model and get coefficients 
% 
%       %avg_train = avg_train';
%       %pos_train = avg_pos_train';
%       avg_train_with_history = avg_train_with_history';
%       pos_train_with_history = pos_train_for_history';
%       
%       % fit model 
%       mdl_x_with_history = fitlm(avg_train_with_history, pos_train_with_history(:, 1)); 
%       mdl_y_with_history = fitlm(avg_train_with_history, pos_train_with_history(:, 2)); 
%       
%       % get model coefficients 
%       model_x_coeffs(:, dir)=mdl_x_with_history.Coefficients.Estimate; 
%       model_y_coeffs(:, dir)=mdl_y_with_history.Coefficients.Estimate; 
      
%% ------ Uncomment this part is for LR without history using rates as input field 

%        % change last input argument to add a neural delay in seconds 
%        [neural_activity, handpos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, training_data, 1, 0.0, 0.001, 0.01);
%         
%        % should not have nans but if so remove 
%        neural_activity_train = remove_nans(neural_activity_train); 
%        handpos_train = remove_nans(handpos_train); 
%         
%        % finds rates 
%        avg_rates=find_rates2(neural_activity_train, 10);
%        avg_pos=find_pos1(handpos_train, 10);
%         
%        rates_train=avg_rates'; 
%        pos_train=avg_pos'; 
%        
%        % fit model
%        mdl_x=fitlm(rates_train, pos_train(:, 1));
%        mdl_y=fitlm(rates_train, pos_train(:, 2));
% 
%        model_x_coeffs(:, dir)=mdl_x.Coefficients.Estimate; 
%        model_y_coeffs(:, dir)=mdl_y.Coefficients.Estimate;  
       
%% ------ Uncomment this part is for LR with history using rates as input field 

%        % change last input argument to add a neural delay in seconds 
%        [neural_activity, handpos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, training_data, 1, 0.0, 0.001, 0.01);
%         
%        % should not have nans but if so remove 
%        neural_activity_train = remove_nans(neural_activity_train); 
%        handpos_train = remove_nans(handpos_train); 
%        
%        % finds rates 
%        avg_rates=find_rates2(neural_activity_train, 10);
%        avg_pos=find_pos1(handpos_train, 10);
%        
%        % adds history to matrices of rates 
%        avg_train_with_history=get_history_matrix(avg_rates, 3);
%        pos_train_with_history = avg_pos(:, 3+2:end);   
%        
%        rates_train_hist=avg_train_with_history'; 
%        pos_train_hist=pos_train_with_history'; 
%        
%        % fit model
%        mdl_x=fitlm(rates_train_hist, pos_train_hist(:, 1));
%        mdl_y=fitlm(rates_train_hist, pos_train_hist(:, 2));
% 
%        model_x_coeffs(:, dir)=mdl_x.Coefficients.Estimate; 
%        model_y_coeffs(:, dir)=mdl_y.Coefficients.Estimate;       
       
%% ------ Uncomment this part is to do PCR using spikes as input 

%       % change last input argument to add a neural delay in seconds 
%       [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train,handpos_validate]=get_io_matrices2(dir, training_data, 1, 0, 0.001, 0.01);
%       neural_activity_train = remove_nans(neural_activity_train); 
%       handpos_train = remove_nans(handpos_train); 
%       
%       % average across trials 
%       avg_train=squeeze(mean(neural_activity_train, 1));
%       avg_pos_train=squeeze(mean(handpos_train, 1));
%       
%       avg_train(isnan(avg_train))=0;
%       avg_pos_train(isnan(avg_pos_train))=0;
%                                                                                                                                                                                           
%       avg_train1 = avg_train';
%       avg_pos_train1 = avg_pos_train';
%       
%       % SVD
%       % input must be ( time x neurons )
%       [U, S, V]=svd(avg_train1);  
%       % find coefficients via PCR 
%       coeff_pcr = V(:, 1:20) / (S(1:20, 1:20)) * (U(:, 1:20))' * avg_pos_train1;
%       
%       % get model coefficients 
%       model_x_coeffs(:, dir)= [mean(squeeze(coeff_pcr(:, 1))); coeff_pcr(:, 1)]; 
%       model_y_coeffs(:, dir)= [mean(squeeze(coeff_pcr(:, 2))); coeff_pcr(:, 2)];

%% ------ Uncomment this part is to do PCR using rates as input 

%       % change last input argument to add a neural delay in seconds 
%       [neural_activity, handpos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, training_data, 1, 0.0, 0.001, 0.01);
%         
%       % should not have nans but if so remove 
%       neural_activity_train = remove_nans(neural_activity_train); 
%       handpos_train = remove_nans(handpos_train); 
%         
%       % finds rates 
%       avg_rates=find_rates2(neural_activity_train, 10);
%       pos_for_rates=find_pos1(handpos_train, 10);
%         
%       rates_train=avg_rates'; 
%       pos_train=pos_for_rates'; 
%       
%       % SVD
%       % input must be ( time x neurons )
%       [U, S, V]=svd(rates_train);  
%       % find coefficients via PCR 
%       coeff_pcr = V(:, 1:20) / (S(1:20, 1:20)) * (U(:, 1:20))' * pos_train;
%       
%       % get model coefficients 
%       model_x_coeffs(:, dir)= [mean(squeeze(coeff_pcr(:, 1))); coeff_pcr(:, 1)]; 
%       model_y_coeffs(:, dir)= [mean(squeeze(coeff_pcr(:, 2))); coeff_pcr(:, 2)];

      %% Uncomment this to perform ridge regression on spikes with history 
%       %change last input argument to add a neural delay in seconds 
%       [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train,handpos_validate]=get_io_matrices2(dir, training_data, 1, 0, 0.001, 0.01);
%       neural_activity_train = remove_nans(neural_activity_train); 
%       handpos_train = remove_nans(handpos_train); 
%       
%       %average across trials 
%       avg_train=squeeze(mean(neural_activity_train, 1));
%       avg_pos_train=squeeze(mean(handpos_train, 1));
%       
%       avg_train(isnan(avg_train))=0;
%       avg_pos_train(isnan(avg_pos_train))=0;
%      
%       %get matrices with/for history
%       avg_train_with_history=get_history_matrix(avg_train, 5);
%       pos_train_for_history = avg_pos_train(:, bins_before+2:end); 
%       
%       %train linear regression model and get coefficients 
% 
%       %avg_train = avg_train';
%       %pos_train = avg_pos_train';
%       avg_train_with_history1 = avg_train_with_history';
%       pos_train_with_history1 = pos_train_for_history';
%     
%       %get model coefficients 
%       model_x_coeffs(:, dir) = ridge(squeeze(pos_train_with_history1(:, 1)), avg_train_with_history1, 1e-4, 0); 
%       model_y_coeffs(:, dir) = ridge(squeeze(pos_train_with_history1(:, 2)), avg_train_with_history1, 1e-4, 0); 

      %% Uncomment this to perform ridge regression on rates with history 
%        % change last input argument to add a neural delay in seconds 
%        [neural_activity, handpos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, training_data, 1, 0.0, 0.001, 0.01);
%         
%        % should not have nans but if so remove 
%        neural_activity_train = remove_nans(neural_activity_train); 
%        handpos_train = remove_nans(handpos_train); 
%         
%        % finds rates 
%        avg_rates=find_rates2(neural_activity_train, 10);
%        position_rates=find_pos1(handpos_train, 10);
% 
%        % adds history to matrices of rates 
%        avg_train_with_history=get_history_matrix(avg_rates, 5);
%        pos_train_with_history = position_rates(:, bins_before+2:end);   
%        
%        rates_train_hist=avg_train_with_history'; 
%        pos_train_hist=pos_train_with_history'; 
%     
%        % get model coefficients 
%        model_x_coeffs(:, dir) = ridge(squeeze(pos_train_hist(:, 1)), rates_train_hist, 1e-4, 0); 
%        model_y_coeffs(:, dir) = ridge(squeeze(pos_train_hist(:, 2)), rates_train_hist, 1e-4, 0);        
    end 
   
  % Return Value:
  modelParameters.Coefficients_x = model_x_coeffs;
  modelParameters.Coefficients_y = model_y_coeffs;
  %modelParameters=struct('Coefficients_x', model_x_coeffs, 'Coefficients_y', model_y_coeffs, 'Classification', classificationMdl2, 'avg', avgTraj); 

  
  %% FUNCTIONS 
  
  % concatenating spiking activity across trials without nan padding 
  function U_matrix=directly_concatrials(dataset, dir, trial_idx_start, trial_idx_end)
    numNeurons=size(dataset(1, dir).spikes, 1); 
    U_first_row=[]; 
    for i=trial_idx_start:trial_idx_end
        U_first_row=[U_first_row dataset(i, dir).spikes(1, :)]; 
    end

    U_matrix=zeros(numNeurons, length(U_first_row)); 
    U_matrix(1, :)=U_first_row; 

    for i=2:numNeurons
        U_first_row=[]; 
        for p=trial_idx_start:trial_idx_end
            U_first_row=[U_first_row dataset(p, dir).spikes(i, :)]; 
        end
        U_matrix(i, :)=U_first_row; 
    end 
  end 
  
  % concatenating position without nan padding 
  function pos =directly_concatrials_pos(dataset, dir, trial_idx_start, trial_idx_end)
    pos_x=[]; 
    pos_y=[]; 
    for i=trial_idx_start:trial_idx_end
        pos_x=[pos_x dataset(i, dir).handPos(1, :)]; 
        pos_y=[pos_y dataset(i, dir).handPos(2, :)]; 
    end 
    pos=[pos_x; pos_y]; 
  end 

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


% finding the position of the hand at intervals dt, 
% which is the product of the firing rate in the interval 
function pos2=find_pos1(handpos, dt)
% input arguments: 
% - handpos of dimensions ( trials x neurons x time )
% - dt multiple of 10

    limit=size(handpos, 3);
    
    % loop to ensure neural activity is in a format multiple of dt 
    for i=dt:dt:limit
        pos_reshaped=handpos(:, :, 1:i); 
    end
    
    lim=size(pos_reshaped, 3)/10; 
    position=squeeze(mean(pos_reshaped, 1));
   
    for i=1:lim
        pos2(:, i)=position(:, i*dt); 
    end
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

% getting matrices of neural activity and position from structure of training data 
% Generate input matrices of K trials x N neurons x T time (set to max duration of a trial with NaN padding)
% Generate output matrices of K trials x 3 spatial dimensions x T time (set to max duration of a trial with NaN padding)

% INPUTS to the FUNCTION: 
% - data structure 
% - direction of interest (which should be the output of the classification process) 
% - fraction of data to train on
% - fraction of data to validate on (the rest will serve as the testing fraction)
% - resolution of data (time binning unit)
% - neural delay in seconds 

% OUTPUTS to the FUNCTION: 
% - neural_activity: matrix of all trials x N neurons x max T time (NaN padding)
% - neural_activity_train, neural_activity_test, neural_activity_validate: fractions of it
% - pos: matrix of all trials x 3 dimensions x max T time (NaN padding)

% Get input and output matrices 
function [neural_activity, pos, neural_activity_train, neural_activity_test, neural_activity_validate, handpos_test,handpos_train, handpos_validate]=get_io_matrices2(dir, data_struct, trainingSplit, validatingSplit, resolution, delay)
      
      bins_correcting=delay/resolution; % shifting matrices by how many bins 
      
      numNeurons = size(data_struct(1,1).spikes, 1); % # neurons
      numTrials = size(data_struct, 1);              % # trials per direction
      numDirections = size(data_struct,2);          % # directions experimented

      % creates array to save the start and end of each trial
      duration = zeros(1,numTrials);          
      for i = 1 : numTrials
          duration(i) = length(data_struct(i,dir).spikes);
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
              activity = data_struct(i,dir).spikes(neuron,320:min_duration-bins_correcting);
              % fills in activity to neural activity matrix
              neural_activity(i, neuron, 1:length(activity) ) = activity; 
          end 
      end 
      
      % Package Hand Data (OUTPUT FROM DECODER)

      % create array to save hand positions (note initially filled w/ NAN)
      pos = zeros(numTrials, 3, min_duration-bins_correcting -320); % note only x-y needed
      for i = 1 : numTrials
          for xyz = 1 : 3
              trajectory = data_struct(i,dir).handPos(xyz,bins_correcting+320:min_duration);
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

function neural_activity = remove_nans(neural_activity)
   for tr=1:size(neural_activity, 1)
      for neuron=1:size(neural_activity, 2)
         for t=1:size(neural_activity, 3)
            if isnan(neural_activity(tr, neuron, t))
               neural_activity(tr, neuron, t)=neural_activity(tr, neuron, t-1); 
            end
         end
      end
   end 
end 

    
end