function cfg = qtp_config_high_spatial2d(datasetFile, frameStoreDir)
% Configuration for MODEL, TRAINING and PREDICTION on already prepared inputs.
% Usage:
%   results = train_qtp_anomaly_cnn_bilstm_bayesopt(cfg);
% Required dataset fields:
%   Train/Val/Test_Input (C-T-N centre inputs), Train/Val/Test_Output (1-T-N),
%   Ps.Input, Ps.Output, Meta, AnomalyInfo and SpatialInfo.FillValues/PatchSize.
% Meta contains FeatureNames, Reference025 and Year/Row/Col for each split.
% Spatial files: frames025_YYYY.mat with FeatureCube(H-W-C-T), FeatureNames,
%   Reference and WindowTokens. Predictors must match the prepared dataset.
% Optional environment variables: QTP_DATASET_FILE and QTP_FRAME_STORE_DIR.
rootDir = fileparts(mfilename('fullpath'));
if nargin<1 || isempty(datasetFile)
    datasetFile = getenv('QTP_DATASET_FILE');
    if isempty(datasetFile)
        datasetFile = fullfile(rootDir,'outputs','dataset','High_spatial2d_seq24_anomaly.mat');
    end
end
if nargin<2 || isempty(frameStoreDir)
    frameStoreDir = getenv('QTP_FRAME_STORE_DIR');
    if isempty(frameStoreDir)
        frameStoreDir = fullfile(fileparts(char(datasetFile)),'spatial025');
    end
end
cfg = struct();
cfg.RootDir = rootDir;
cfg.RandomSeed = 42;
cfg.Data.DatasetFile = char(datasetFile);
cfg.Data.DatasetDir = fileparts(char(datasetFile));
cfg.Data.OutputDir = fullfile(rootDir,'outputs');
cfg.Data.ModelDir = fullfile(cfg.Data.OutputDir,'models');
cfg.Data.ModelFile = fullfile(cfg.Data.ModelDir,'High_CNN2D_BiLSTM_Attention_HydroSignal_v2_R2.mat');
cfg.Dataset.NumFeatures = 11;
cfg.Dataset.SequenceLength = 24;
cfg.Dataset.Cast = 'single';
% Actual split and preprocessing history are read from the prepared Meta.
cfg.Spatial.Enabled = true;
cfg.Spatial.PatchSize = [5 5];
cfg.Spatial.Padding = "replicate";
cfg.Spatial.MissingNeighbourPolicy = "training_channel_mean";
cfg.Spatial.MaxCachedYears = 20;
cfg.Spatial.FrameStoreDir = char(frameStoreDir);
cfg.Model.SpatialKernelSize = [3 3];
cfg.Model.SpatialKernelSize2 = [3 3];
cfg.Model.CnnFcUnits = 32;
cfg.Model.LstmFcUnits = 32;
cfg.Model.DefaultGaussianStd = 0.05;
cfg.Model.DefaultHuberDelta = 1.0;
cfg.Model.UseCNNAttention = true;
cfg.Model.UseSpatioTemporalAttention = true;
cfg.Model.LossType = "hydro_signal_huber";
cfg.Model.BiasPenaltyWeight = 0.008;
cfg.Model.CorrelationPenaltyWeight = 0.18;
cfg.Model.StdPenaltyWeight = 0.30;
cfg.Model.DifferencePenaltyWeight = 0.08;
cfg.Model.AnnualDifferencePenaltyWeight = 0.06;

cfg.HydroObjective.RMSEWeight = 0.85;
cfg.HydroObjective.R2Weight = 8.0;
cfg.HydroObjective.R2Target = 0.74;
cfg.HydroObjective.CorrelationWeight = 2.5;
cfg.HydroObjective.StdRatioWeight = 2.0;
cfg.HydroObjective.DifferenceWeight = 0.35;
cfg.HydroObjective.AnnualDifferenceWeight = 0.25;

cfg.BayesOpt.MaxObjectiveEvaluations = 35;
cfg.BayesOpt.InitialEpochs = 35;
cfg.BayesOpt.FinalEpochs = 140;
cfg.BayesOpt.MiniBatchSize = 256;
cfg.BayesOpt.ValidationFrequency = 25;
cfg.BayesOpt.VerboseFrequency = 25;
cfg.BayesOpt.ExecutionEnvironment = "auto";
cfg.BayesOpt.UseParallel = false;
cfg.BayesOpt.OptimVars = [ ...
    optimizableVariable('NumHiddenUnits',[96 192],'Type','integer')
    optimizableVariable('LearnRate',[2e-4 9e-4],'Transform','log')
    optimizableVariable('NumFilters',[128 256],'Type','integer')
    optimizableVariable('DropoutRate',[0.05 0.16])];
cfg.Prediction.BatchSize = 64;
cfg.Prediction.MonteCarloSamples = 20;
cfg.Prediction.MonteCarloSamplesEval = 8;
end
