function results = recover(results)
	%% Set Variables
	% For you, it will be something like ../data
	DATA_PATH = '/home/chris/JLPeacock_JNeuro2008/orig/mat';
	N_SUB = 10;
	N_CV = 10;
	% Just an estimate for the max number of iterations, for preallocating.
    TargetCategory = 'TrueFaces';
    opts = glmnetSet();
	opts.alpha = 0; % 1 means LASSO; 0 means Ridge

	%% Set Y and CVBLOCKS
	load(fullfile(DATA_PATH,'jlp_metadata.mat'));
	Y = {metadata.(TargetCategory)};
	CVBLOCKS = {metadata.CVBLOCKS};

	%% Load the data for subject 1
	load(fullfile(DATA_PATH,'jlp01.mat'),'X');

	%% Subset CV and Y for just subject 1
	CVBLOCKS = CVBLOCKS{1};
	Y = Y{1};

	start_cc = 1;

	errU = zeros(1,N_CV);
	dpU = zeros(1,N_CV);
%     fitObj_cv(10) = struct();
%     fitObj(10) = struct();

UNUSED_VOXELS = results.UNUSED_VOXELS;
% 	load('results_subj1.mat');
% 	UNUSED_VOXELS = true(size(results.fitObj(1,1).beta,1),N_CV);
% 	for i=1:2
% 		for j=1:10
% 			UNUSED_VOXELS(UNUSED_VOXELS(:,j),j) = results.fitObj(i,j).beta == 0;
% 		end
%     end

    dp_c_glm = zeros(1,N_CV);
    err_c_glm = zeros(1,N_CV);
    ix = 1:N_CV;
	for cc = start_cc:N_CV
        disp(cc)
		% Remove the holdout set
		FINAL_HOLDOUT = CVBLOCKS(:,cc);
        CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc);
		Xtrain = X(~FINAL_HOLDOUT,:);
		Ytrain = Y(~FINAL_HOLDOUT);
		
		%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
		USEFUL_VOXELS = ~UNUSED_VOXELS(:,:,end-2);
        USEFUL_VOXELS = all([USEFUL_VOXELS(:,cc),sum(USEFUL_VOXELS(:,ix~=cc),2)>2],2);
%         USEFULL_VOXELS = USEFULL_VOXELS(:,cc);
        
		b=glmfit(Xtrain(:,USEFUL_VOXELS),Ytrain,'binomial');
		a0 = b(1);
		b(1) = [];
	
		% Evaluate this new model on the holdout set.
		% Step 1: compute the model predictions.
		yhat = X(:,USEFUL_VOXELS)*b + a0;
		% Step 2: compute the error of those predictions.
		err_c_glm(1,cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
		% Step 3: compute the sensitivity of those predictions (dprime).
		dp_c_glm(1,cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
        
        % Convert CV2 to fold_id
        fold_id = sum(bsxfun(@times,double(CV2),1:9),2);

        % For some reason, this must be a row vector.
        fold_id = transpose(fold_id);

        % Run cvglmnet to determine a good lambda.
        fitObj_cv(cc) = cvglmnet(Xtrain(:,USEFUL_VOXELS),Ytrain, ...
                                 'binomial',opts,'class',9,fold_id);

        % Set that lambda in the opts structure, and fit a new model.
        opts.lambda = fitObj_cv(cc).lambda_min;
        fitObj(cc) = glmnet(Xtrain(:,USEFUL_VOXELS),Ytrain,'binomial',opts);

        % Unset lambda, so next time around cvglmnet will look for lambda
        % itself.
        opts = rmfield(opts,'lambda');

        % Evaluate this new model on the holdout set.
        % Step 1: compute the model predictions.
        yhat = (X(:,USEFUL_VOXELS)*fitObj(cc).beta)+fitObj(cc).a0;
        % Step 2: compute the error of those predictions.
        errU(cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
        % Step 3: compute the sensitivity of those predictions (dprime).
        dpU(cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
	end
	%% Package results 
    results.err_c_glm = err_c_glm;
    results.dp_c_glm = dp_c_glm;
	results.errU = errU;
	results.dpU = dpU;
    results.UNUSED_VOXELS = UNUSED_VOXELS;
end
