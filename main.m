% Notes: 
% We want to stop iterating when the mean over cross validations runs is not
% significantly different from chance.
function results = main()
	%% Set Variables
	% For you, it will be something like ../data
% 	DATA_PATH = '/home/chris/JLPeacock_JNeuro2008/orig/mat';
    DATA_PATH = '../data';
% 	N_SUB = 10;
	N_CV = 10;
	% Just an estimate for the max number of iterations, for preallocating.
	N_ITER_EST = 10; 
    TargetCategory = 'TrueFaces';
	opts = glmnetSet();
	opts.alpha = 1; % 1 means LASSO; 0 means Ridge

	%% Set Y and CVBLOCKS
	load(fullfile(DATA_PATH,'jlp_metadata.mat'));
	Y = {metadata.(TargetCategory)};
	CVBLOCKS = {metadata.CVBLOCKS};

	%% Load the data for subject 1
	load(fullfile(DATA_PATH,'jlp01.mat'),'X');

	%% Subset CV and Y for just subject 1
	CVBLOCKS = CVBLOCKS{1};
	Y = Y{1};

	%% Before starting the loop: see if there is a checkpoint file.
	if exist(fullfile(pwd, 'CHECKPOINT.mat'), 'file') == 2
		load('CHECKPOINT.mat');
		start_cc = cc;
		fprintf('++Resuming from CV%02d\n',cc);
	else
		fprintf('starting from scratch\n');
		ii = 0;
        jj = 0;
		start_cc = 1;
		dp = zeros(N_ITER_EST,N_CV);
		err = zeros(N_ITER_EST,N_CV);
		fitObj = init_glmnet_result_struct('glmnet',[N_ITER_EST, N_CV]);
        fitObj_cv = init_glmnet_result_struct('cvglmnet',[N_ITER_EST, N_CV]);
        UNUSED_VOXELS = true(size(X,2),N_CV,N_ITER_EST);
    end
	
    %% Start the loop
	while true
		% Increment loop counter
		ii = ii + 1;

		% Setup a loop over holdout sets.
		for cc = start_cc:N_CV
			fprintf('cv%02d\n',cc);
			% Pick a final holdout set
			FINAL_HOLDOUT = CVBLOCKS(:,cc);

			% Remove the holdout set
			CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc); 
			Xtrain = X(~FINAL_HOLDOUT,:);
			Ytrain = Y(~FINAL_HOLDOUT);

			% Convert CV2 to fold_id
			fold_id = sum(bsxfun(@times,double(CV2),1:9),2);

			% For some reason, this must be a row vector.
			fold_id = transpose(fold_id);

			% Run cvglmnet to determine a good lambda.
			fitObj_cv(ii,cc) = cvglmnet(Xtrain(:,UNUSED_VOXELS(:,cc,ii)),Ytrain, ...
				                     'binomial',opts,'class',9,fold_id);
            
			% Set that lambda in the opts structure, and fit a new model.
			opts.lambda = fitObj_cv(ii,cc).lambda_min;
  			fitObj(ii,cc) = glmnet(Xtrain(:,UNUSED_VOXELS(:,cc,ii)),Ytrain,'binomial',opts);
            
			% Unset lambda, so next time around cvglmnet will look for lambda
			% itself.
			opts = rmfield(opts,'lambda');

			% Evaluate this new model on the holdout set.
			% Step 1: compute the model predictions.
			yhat = (X(:,UNUSED_VOXELS(:,cc,ii))*fitObj(ii,cc).beta)+fitObj(ii,cc).a0;
			% Step 2: compute the error of those predictions.
			err(ii,cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
			% Step 3: compute the sensitivity of those predictions (dprime).
			dp(ii,cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
			fprintf('% 1.3f% 1.3f\n',err(ii,cc),dp(ii,cc));
			
			% Indicate which voxels were used/update the set of unused voxels.
            UNUSED_VOXELS(:,cc,ii+1) = UNUSED_VOXELS(:,cc,ii);
			UNUSED_VOXELS(UNUSED_VOXELS(:,cc,ii),cc,ii+1) = fitObj(ii,cc).beta==0;

			% Save a checkpoint file
			save('CHECKPOINT.mat','cc','UNUSED_VOXELS','ii','jj','err','dp','fitObj','fitObj_cv');

        end
        
        %% Test if the dprime is significantly greater than zero.
		h = ttest(dp(ii,:),0,'Alpha',0.05,'Tail','right');
		if isnan(h)
			h = false;
		end
        if h==true
            % If it is, reset jj ...
            jj = 0;
        else
            % If it is not, increment j ...
            jj = jj + 1;
        end
        
        % Because if it is not significant for more than two iterations,
        % then we should stop iterating.
        if jj > 2;
            break
        end
		
		fprintf('\n');
    end

	%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
	USEFUL_VOXELS = ~UNUSED_VOXELS(:,:,ii-3);
    USEFUL_VOXELS = any(USEFUL_VOXELS,3);

    opts.alpha = 0; % ridge regression
    fitObj_ridge = init_glmnet_result_struct('glmnet',[1, N_CV]);
    fitObj_cv_ridge = init_glmnet_result_struct('cvglmnet',[1, N_CV]);
    for cc = start_cc:N_CV
        disp(cc)
		% Remove the holdout set
		FINAL_HOLDOUT = CVBLOCKS(:,cc);
        CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc);
		Xtrain = X(~FINAL_HOLDOUT,:);
		Ytrain = Y(~FINAL_HOLDOUT);
		        
        % Convert CV2 to fold_id
        fold_id = sum(bsxfun(@times,double(CV2),1:9),2);

        % For some reason, this must be a row vector.
        fold_id = transpose(fold_id);

        % Run cvglmnet to determine a good lambda.
        fitObj_cv_ridge(cc) = cvglmnet(Xtrain(:,USEFUL_VOXELS),Ytrain, ...
                                 'binomial',opts,'class',9,fold_id);

        % Set that lambda in the opts structure, and fit a new model.
        opts.lambda = fitObj_cv_ridge(cc).lambda_min;
        fitObj_ridge(cc) = glmnet(Xtrain(:,USEFUL_VOXELS),Ytrain,'binomial',opts);

        % Unset lambda, so next time around cvglmnet will look for lambda
        % itself.
        opts = rmfield(opts,'lambda');

        % Evaluate this new model on the holdout set.
        % Step 1: compute the model predictions.
        yhat = (X(:,USEFUL_VOXELS)*fitObj_ridge(cc).beta)+fitObj(cc).a0;
        % Step 2: compute the error of those predictions.
        err_ridge(cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
        % Step 3: compute the sensitivity of those predictions (dprime).
        dp_ridge(cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
	end
	%% Package results 
	results.errU = err_ridge;
	results.dpU = dp_ridge;
    results.UNUSED_VOXELS = UNUSED_VOXELS;
    results.fitObj_ridge = fitObj_ridge;
    results.fitObj_cv_ridge = fitObj_ridge_cv;
    results.fitObj = fitObj(1:ii,:);
    results.fitObj_cv = fitObj_cv(1:ii,:);
    results.err = err(1:ii,:);
    results.dp = dp(1:ii,:);
	results.errU = err_ridge;
	results.dpU = dp_ridge;
	results.UNUSED_VOXELS = UNUSED_VOXELS(:,:,1:ii);
    delete('CHECKPOINT.mat');
end
