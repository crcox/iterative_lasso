% You may find yourself writing several functions for this project. If so,
% main can be used to call them all...
%interative lasso
function [] = main()
    for cc = 1:10
        currentCV = metadata.CVBLOCKS(:,cc);
        Xtrain = X(~currentCV,:);        
        fitObj = cvglmnet(Ytrain, Xtrain);
        
        fitObjFinal = glmnet(Ytrain, Xtrain); 
                
        %Evaluate on current CV portion of X and Y.
        %Track the non-zero elements of the beta solution. 
    end
end