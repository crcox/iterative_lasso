% You may find yourself writing several functions for this project. If so,
% main can be used to call them all...
%interative lasso
function [] = main()
	%% Set Variables
	% For you, it will be something like ../data
	DATA_PATH = '/home/chris/JLPeacock_JNeuro2008/orig/mat';
	N_SUB = 10;
	N_CV = 10;
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

	% Before starting the loop: see if there is a checkpoint file.
	if exist(fullfile(pwd, 'CHECKPOINT.mat'), 'file') == 2
		load('CHECKPOINT.mat','cc');
		start_cc = cc;
		fprintf('++Resuming from CV%02d\n',cc);
	else
		start_cc = 1;
	end

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
		

		% Before starting the loop: see if there is a checkpoint file.
		if exist(fullfile(pwd, 'CHECKPOINT.mat'), 'file') == 2
			load('CHECKPOINT.mat','UNUSED_VOXELS','ii','err','dp');
		else
			UNUSED_VOXELS = true(size(X,2),1);
			ii = 0;
			err = 0;
			dp = 0;
		end

		% Setup a loop for eliminating voxels
		fprintf('\t% 6s% 6s% 6s\n','vox','err','dp');
		while any(UNUSED_VOXELS)
			fprintf('\t% 6d',sum(UNUSED_VOXELS));
			% Increment loop counter
			ii = ii + 1;

			% Run cvglmnet to determine a good lambda.
			fitObj_cv(ii) = cvglmnet(Xtrain(:,UNUSED_VOXELS),Ytrain, ...
				                     'binomial',opts,'class',9,fold_id);

			% Set that lambda in the opts structure, and fit a new model.
			opts.lambda = fitObj_cv(ii).lambda_min;
			fitObj(ii) = glmnet(Xtrain(:,UNUSED_VOXELS),Ytrain,'binomial',opts);

			% Unset lambda, so next time around cvglmnet will look for lambda
			% itself.
			opts = rmfield(opts,'lambda');

			% Evaluate this new model on the holdout set.
			% Step 1: compute the model predictions.
			yhat = (X(:,UNUSED_VOXELS)*fitObj(ii).beta)+fitObj.a0;
			% Step 2: compute the error of those predictions.
			err(ii,cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
			% Step 3: compute the sensitivity of those predictions (dprime).
			dp(ii,cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
			fprintf('% 1.3f% 1.3f\n',err,dp);

			% Indicate which voxels were used/update the set of unused voxels.
			UNUSED_VOXELS(UNUSED_VOXELS) = fitObj(ii).beta==0;
			save('CHECKPOINT.mat','cc','UNUSED_VOXELS','ii','err','dp');
		end
		fprintf('\n');
	end
end
