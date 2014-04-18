function results = recover()
	%% Set Variables
	% For you, it will be something like ../data
	DATA_PATH = '/home/chris/JLPeacock_JNeuro2008/orig/mat';
	N_SUB = 10;
	N_CV = 10;
	% Just an estimate for the max number of iterations, for preallocating.
    TargetCategory = 'TrueFaces';

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

	load('results_subj1.mat');
	UNUSED_VOXELS = true(size(results.fitObj(1,1).beta,1),N_CV);
	for i=1:2
		for j=1:10
			UNUSED_VOXELS(UNUSED_VOXELS(:,j),j) = results.fitObj(i,j).beta == 0;
		end
	end

	for cc = start_cc:N_CV
		% Remove the holdout set
		FINAL_HOLDOUT = CVBLOCKS(:,cc);
		Xtrain = X(~FINAL_HOLDOUT,:);
		Ytrain = Y(~FINAL_HOLDOUT);
		
		%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
		USEFUL_VOXELS = ~UNUSED_VOXELS(:,cc,end);
		b=glmfit(Xtrain(:,USEFUL_VOXELS),Ytrain,'binomial');
		a0 = b(1);
		b(1) = [];
	
		% Evaluate this new model on the holdout set.
		% Step 1: compute the model predictions.
		yhat = X(:,USEFUL_VOXELS)*b + a0;
		% Step 2: compute the error of those predictions.
		errU(1,cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
		% Step 3: compute the sensitivity of those predictions (dprime).
		dpU(1,cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
	end
	%% Package results 
	results.errU = errU;
	results.dpU = dpU;
end
