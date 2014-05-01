function CVerr = init_glmnet_result_struct(function_name,dimensions,varargin)
d1 = dimensions(1);
d2 = dimensions(2);
if nargin>2
    keep = varargin{1};
else
    keep = false;
end
switch function_name
    case 'cvglmnet'
        CVerr(d1,d2).lambda = [];
        CVerr(d1,d2).cvm = []; CVerr(d1,d2).cvsd = []; 
        CVerr(d1,d2).cvup = []; CVerr(d1,d2).cvlo = []; CVerr(d1,d2).nzero = [];
        CVerr(d1,d2).name = []; CVerr(d1,d2).glmnet_fit = [];
        if keep
            CVerr(d1,d2).fit_preval = []; CVerr(d1,d2).foldid = [];
        end
        CVerr(d1,d2).lambda_min = [];
        CVerr(d1,d2).lambda_1se = [];
        CVerr(d1,d2).class = 'cv.glmnet';
    case 'glmnet'
        CVerr(d1,d2).a0 = [];
        CVerr(d1,d2).beta = [];
        CVerr(d1,d2).dev = [];
        CVerr(d1,d2).nulldev = [];
        CVerr(d1,d2).df = [];
        CVerr(d1,d2).lambda = [];
        CVerr(d1,d2).npasses = [];
        CVerr(d1,d2).jerr = [];
        CVerr(d1,d2).dim = [];
        CVerr(d1,d2).offset = [];
        CVerr(d1,d2).class = [];
end
