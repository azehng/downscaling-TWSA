function results = train_qtp_anomaly_cnn_bilstm_bayesopt(cfg)
%TRAIN_QTP_ANOMALY_CNN_BILSTM_BAYESOPT Train CNN-BiLSTM-Attention on TWSA anomalies.

if nargin == 0
    cfg = qtp_config_high_spatial2d();
end

rng(cfg.RandomSeed, 'twister');
iEnsureDir(cfg.Data.ModelDir);

if ~isfile(cfg.Data.DatasetFile)
    error('train_anomaly:MissingDataset', ...
        'Prepared anomaly dataset not found: %s. Configure cfg.Data.DatasetFile.', cfg.Data.DatasetFile);
end

S = load(cfg.Data.DatasetFile, ...
    'Train_Input', 'Train_Output', ...
    'Val_Input', 'Val_Output', ...
    'Test_Input', 'Test_Output', ...
    'Ps', 'Meta', 'AnomalyInfo', 'SpatialInfo');

DatasetSignature = iDatasetSignature(cfg.Data.DatasetFile);
Ps = S.Ps;
Meta = S.Meta;
AnomalyInfo = S.AnomalyInfo;

fprintf('Training anomaly model.\n');
fprintf('Train samples: %d\n', size(S.Train_Input, 3));
fprintf('Val samples: %d\n', size(S.Val_Input, 3));
fprintf('Test samples: %d\n', size(S.Test_Input, 3));
fprintf('Dataset file: %s\n', cfg.Data.DatasetFile);
fprintf('Model file: %s\n', cfg.Data.ModelFile);
fprintf('LossType=%s | BiasWeight=%.4f | CorrWeight=%.4f\n', ...
    string(cfg.Model.LossType), cfg.Model.BiasPenaltyWeight, cfg.Model.CorrelationPenaltyWeight);

cfg.Dataset.NumFeatures = numel(Meta.FeatureNames);
if isfield(S, 'SpatialInfo')
    SpatialInfo = S.SpatialInfo;
else
    error('spatial2d:OldDataset', 'Prepared CNN2D inputs must contain SpatialInfo; temporal-only inputs cannot supply spatial neighbours.');
end
assert(isequal(cfg.Spatial.PatchSize, SpatialInfo.PatchSize), ...
    'Patch size differs from the prepared dataset configuration.');
fillValues = SpatialInfo.FillValues;
trainSource = QTPSpatialPatchSource(Meta.Train, cfg.Spatial.FrameStoreDir, 'frames025', Ps.Input, fillValues, cfg, Meta.FeatureNames, Meta.Reference025);
valSource = QTPSpatialPatchSource(Meta.Val, cfg.Spatial.FrameStoreDir, 'frames025', Ps.Input, fillValues, cfg, Meta.FeatureNames, Meta.Reference025);
testSource = QTPSpatialPatchSource(Meta.Test, cfg.Spatial.FrameStoreDir, 'frames025', Ps.Input, fillValues, cfg, Meta.FeatureNames, Meta.Reference025);
trainIndices = (1:trainSource.NumObservations)';
trainXCell = QTPSpatialPatchDatastore(trainSource,S.Train_Output,trainIndices,cfg.BayesOpt.MiniBatchSize);
fprintf('Training observations per epoch: %d\n', numel(trainIndices));
valXCell = QTPSpatialPatchDatastore(valSource,S.Val_Output,(1:valSource.NumObservations)',cfg.BayesOpt.MiniBatchSize);
trainYCell = [];
valYCell = [];
objective = @(optVars) iObjective(optVars, cfg, trainXCell, trainYCell, ...
    valSource, S.Val_Output, valXCell, valYCell, S.Ps.Output);

fprintf('Starting Bayesian optimization with %d evaluations...\n', cfg.BayesOpt.MaxObjectiveEvaluations);
bayesResults = bayesopt(objective, cfg.BayesOpt.OptimVars, ...
    'MaxObjectiveEvaluations', cfg.BayesOpt.MaxObjectiveEvaluations, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', cfg.BayesOpt.UseParallel, ...
    'Verbose', 1, ...
    'PlotFcn', []);

if ~any(isfinite(bayesResults.ObjectiveTrace) & bayesResults.ObjectiveTrace < 1e6)
    error('spatial2d:AllTrialsFailed', 'All Bayesian trials failed. Inspect warnings before continuing.');
end
bestParams = iTableRowToStruct(bayesResults.XAtMinObjective);
fprintf('Best hyperparameters:\n');
disp(bestParams);

fprintf('Starting final training for %d epochs...\n', cfg.BayesOpt.FinalEpochs);
[net, finalTrainInfo] = iTrainModel(cfg, bestParams, ...
    trainXCell, trainYCell, valXCell, valYCell, cfg.BayesOpt.FinalEpochs);

trainMetrics = iEvaluateSplit(net, cfg, trainSource, S.Train_Output, S.Ps.Output);
valMetrics = iEvaluateSplit(net, cfg, valSource, S.Val_Output, S.Ps.Output);
testMetrics = iEvaluateSplit(net, cfg, testSource, S.Test_Output, S.Ps.Output);

Metrics = struct2table([ ...
    iMetricsRow("train", trainMetrics)
    iMetricsRow("val", valMetrics)
    iMetricsRow("test", testMetrics)]);
Metrics.TargetMode = repmat("TWSA_monthly_anomaly", height(Metrics), 1);

save(cfg.Data.ModelFile, ...
    'net', 'bestParams', 'bayesResults', 'finalTrainInfo', ...
    'Metrics', 'cfg', 'Ps', 'Meta', 'AnomalyInfo', 'SpatialInfo', 'DatasetSignature', '-v7.3');
writetable(Metrics, fullfile(cfg.Data.ModelDir, 'High_spatial2d_metrics.csv'));

results = struct();
results.net = net;
results.bestParams = bestParams;
results.bayesResults = bayesResults;
results.finalTrainInfo = finalTrainInfo;
results.Metrics = Metrics;
results.Meta = Meta;
results.Ps = Ps;
results.AnomalyInfo = AnomalyInfo;
results.DatasetSignature = DatasetSignature;

disp(Metrics);
fprintf('Anomaly model saved to %s\n', cfg.Data.ModelFile);
end

function objective = iObjective(optVars, cfg, trainXCell, trainYCell, valX, valY, valXCell, valYCell, outputPs)
params = iTableRowToStruct(optVars);
rng(cfg.RandomSeed, 'twister');

try
    [net, ~] = iTrainModel(cfg, params, trainXCell, trainYCell, valXCell, valYCell, cfg.BayesOpt.InitialEpochs);
    [predNorm, ~] = qtp_mc_predict(net, valX, max(3, min(cfg.Prediction.MonteCarloSamplesEval, 5)), cfg.Prediction.BatchSize);
    pred = iReverseNormalize(predNorm, outputPs);
    truth = iReverseNormalize(valY, outputPs);
    metrics = qtp_regression_metrics(truth, pred);
    objective = iHydroObjective(metrics, truth, pred, cfg);
catch ME
    warning('train_anomaly:BayesObjectiveFailed', '%s', ME.message);
    objective = 1e6;
end
end

function [net, trainInfo] = iTrainModel(cfg, params, trainXCell, trainYCell, valXCell, valYCell, numEpochs)
lgraph = qtp_build_network(cfg, params);

options = trainingOptions('adam', ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', cfg.BayesOpt.MiniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', params.LearnRate, ...
    'ValidationData', valXCell, ...
    'ValidationFrequency', cfg.BayesOpt.ValidationFrequency, ...
    'ExecutionEnvironment', char(cfg.BayesOpt.ExecutionEnvironment), ...
    'Verbose', true, ...
    'VerboseFrequency', cfg.BayesOpt.VerboseFrequency, ...
    'Plots', 'none');

reset(trainXCell);
reset(valXCell);
[net, trainInfo] = trainNetwork(trainXCell, lgraph, options);
end

function metrics = iEvaluateSplit(net, cfg, X, Y, outputPs)
[predNorm, predStdNorm] = qtp_mc_predict(net, X, cfg.Prediction.MonteCarloSamplesEval, cfg.Prediction.BatchSize);
pred = iReverseNormalize(predNorm, outputPs);
truth = iReverseNormalize(Y, outputPs);
predStd = iReverseNormalize(predStdNorm, outputPs, true);

metrics = qtp_regression_metrics(truth, pred);
metrics.UncertaintyMean = mean(predStd(:), 'omitnan');
metrics.SampleCount = X.NumObservations;
end

function out = iReverseNormalize(data, ps, isStd)
if nargin < 3
    isStd = false;
end

if ~isStd
    flat = reshape(double(data), size(data, 1), []);
    flat = mapminmax('reverse', flat, ps);
    out = reshape(flat, size(data));
    return;
end

gain = ps.gain(:);
flat = reshape(double(data), size(data, 1), []);
out = reshape(flat ./ gain, size(data));
end

function row = iMetricsRow(splitName, metrics)
row = struct();
row.Split = string(splitName);
row.RMSE = metrics.RMSE;
row.MAE = metrics.MAE;
row.MBE = metrics.MBE;
row.R2 = metrics.R2;
row.CC = metrics.CC;
row.UncertaintyMean = metrics.UncertaintyMean;
row.SampleCount = metrics.SampleCount;
end

function objective = iHydroObjective(metrics, truth, pred, cfg)
rmseWeight = iGetNestedField(cfg, {'HydroObjective', 'RMSEWeight'}, 1.0);
corrWeight = iGetNestedField(cfg, {'HydroObjective', 'CorrelationWeight'}, 2.0);
r2Weight = iGetNestedField(cfg, {'HydroObjective', 'R2Weight'}, 0.0);
r2Target = iGetNestedField(cfg, {'HydroObjective', 'R2Target'}, 0.0);
stdWeight = iGetNestedField(cfg, {'HydroObjective', 'StdRatioWeight'}, 1.0);
diffWeight = iGetNestedField(cfg, {'HydroObjective', 'DifferenceWeight'}, 0.25);
annualDiffWeight = iGetNestedField(cfg, {'HydroObjective', 'AnnualDifferenceWeight'}, 0.0);

truthFlat = reshape(double(truth), [], size(truth, ndims(truth)));
predFlat = reshape(double(pred), [], size(pred, ndims(pred)));

truthCentered = truthFlat - mean(truthFlat, 1, 'omitnan');
predCentered = predFlat - mean(predFlat, 1, 'omitnan');
truthStd = sqrt(mean(truthCentered .^ 2, 1, 'omitnan'));
predStd = sqrt(mean(predCentered .^ 2, 1, 'omitnan'));
stdRatio = predStd ./ max(truthStd, eps);
stdPenalty = mean((stdRatio - 1) .^ 2, 'omitnan');

if size(truthFlat, 1) >= 2
    truthDiff = diff(truthFlat, 1, 1);
    predDiff = diff(predFlat, 1, 1);
    diffPenalty = sqrt(mean((predDiff - truthDiff) .^ 2, 'all', 'omitnan'));
else
    diffPenalty = 0;
end

if size(truthFlat, 1) >= 13
    truthAnnualDiff = truthFlat(13:end, :) - truthFlat(1:end-12, :);
    predAnnualDiff = predFlat(13:end, :) - predFlat(1:end-12, :);
    annualDiffPenalty = sqrt(mean((predAnnualDiff - truthAnnualDiff) .^ 2, 'all', 'omitnan'));
else
    annualDiffPenalty = 0;
end

corrPenalty = 1 - metrics.CC;
r2Penalty = max(0, r2Target - metrics.R2);
if ~isfinite(corrPenalty)
    corrPenalty = 1;
end
if ~isfinite(r2Penalty)
    r2Penalty = 1;
end
if ~isfinite(stdPenalty)
    stdPenalty = 1;
end
if ~isfinite(diffPenalty)
    diffPenalty = metrics.RMSE;
end
if ~isfinite(annualDiffPenalty)
    annualDiffPenalty = metrics.RMSE;
end

objective = rmseWeight .* metrics.RMSE + ...
    corrWeight .* corrPenalty + ...
    r2Weight .* r2Penalty + ...
    stdWeight .* stdPenalty + ...
    diffWeight .* diffPenalty + ...
    annualDiffWeight .* annualDiffPenalty;
end

function value = iGetNestedField(s, names, defaultValue)
value = defaultValue;
cursor = s;
for k = 1:numel(names)
    name = names{k};
    if ~isstruct(cursor) || ~isfield(cursor, name) || isempty(cursor.(name))
        return;
    end
    cursor = cursor.(name);
end
value = cursor;
end

function params = iTableRowToStruct(optVars)
if istable(optVars)
    params = table2struct(optVars(1, :));
else
    params = optVars;
end
end

function iEnsureDir(folder)
if ~exist(folder, 'dir')
    mkdir(folder);
end
end

function signature = iDatasetSignature(filePath)
info = dir(filePath);
signature = struct();
signature.FilePath = string(filePath);
signature.Bytes = info.bytes;
signature.Datenum = info.datenum;
signature.Date = string(info.date);
end
