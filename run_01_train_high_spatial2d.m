clear; clc;
packageRoot = fileparts(mfilename('fullpath'));
cd(packageRoot);
addpath(packageRoot);
% Configure QTP_DATASET_FILE/QTP_FRAME_STORE_DIR, or supply the two paths below.
cfg = qtp_config_high_spatial2d();
results = train_qtp_anomaly_cnn_bilstm_bayesopt(cfg); %#ok<NASGU>
