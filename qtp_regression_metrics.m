function metrics = qtp_regression_metrics(trueVals, predVals)
%QTP_REGRESSION_METRICS Compute scalar regression metrics.

trueVals = double(trueVals(:));
predVals = double(predVals(:));

valid = isfinite(trueVals) & isfinite(predVals);
trueVals = trueVals(valid);
predVals = predVals(valid);

if isempty(trueVals)
    metrics = struct('RMSE', NaN, 'MAE', NaN, 'MBE', NaN, 'R2', NaN, 'CC', NaN);
    return;
end

residual = predVals - trueVals;
ssRes = sum(residual .^ 2);
ssTot = sum((trueVals - mean(trueVals)) .^ 2);

metrics = struct();
metrics.RMSE = sqrt(mean(residual .^ 2));
metrics.MAE = mean(abs(residual));
metrics.MBE = mean(residual);
if ssTot > 0
    metrics.R2 = 1 - ssRes / ssTot;
else
    metrics.R2 = NaN;
end
if numel(trueVals) > 1
    cc = corrcoef(trueVals, predVals, 'Rows', 'complete');
    metrics.CC = cc(1, 2);
else
    metrics.CC = NaN;
end
end
