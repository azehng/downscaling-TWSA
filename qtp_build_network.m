function lgraph = qtp_build_network(cfg, params)
%QTP_BUILD_NETWORK Spatial 2D CNN and centre-pixel BiLSTM, sequence-to-sequence.

arguments
    cfg (1,1) struct
    params (1,1) struct
end

numFeatures = cfg.Dataset.NumFeatures;
cnnUnits = cfg.Model.CnnFcUnits;
lstmUnits = cfg.Model.LstmFcUnits;

patchSize = cfg.Spatial.PatchSize;
inputLayer = sequenceInputLayer([patchSize, numFeatures], ...
    'Name', 'input', 'Normalization', 'none');
foldLayer = sequenceFoldingLayer('Name', 'fold');
noiseLayer = GaussianNoiseLayer(cfg.Model.DefaultGaussianStd, 'input_noise');
cnnBranch = [
    convolution2dLayer(cfg.Model.SpatialKernelSize, params.NumFilters, Padding='same', Name='cnn_conv1')
    reluLayer(Name='cnn_relu1')
    layerNormalizationLayer(Name='cnn_norm1')
    MonteCarloDropoutLayer(params.DropoutRate, 'cnn_dropout1')
    convolution2dLayer(cfg.Model.SpatialKernelSize2, max(4, floor(params.NumFilters/2)), Padding='same', Name='cnn_conv2')
    reluLayer(Name='cnn_relu2')
    layerNormalizationLayer(Name='cnn_norm2')
    ];
if cfg.Model.UseCNNAttention
    cnnBranch = [cnnBranch; SpatialChannelAttention2DLayer('cnn_attention')];
end
cnnBranch = [cnnBranch
    globalAveragePooling2dLayer(Name='cnn_spatial_pool')
    sequenceUnfoldingLayer(Name='unfold_cnn')
    flattenLayer(Name='cnn_flatten')
    fullyConnectedLayer(cnnUnits, Name='cnn_fc')];
lstmBranch = [
    CenterPixel2DLayer('center_pixel')
    sequenceUnfoldingLayer(Name='unfold_lstm')
    flattenLayer(Name='center_flatten')
    bilstmLayer(params.NumHiddenUnits, OutputMode='sequence', Name='bilstm')
    MonteCarloDropoutLayer(params.DropoutRate, 'lstm_dropout')
    fullyConnectedLayer(lstmUnits, Name='lstm_fc')];

mergeBranch = [
    concatenationLayer(1, 2, Name='concat')
    ];

if ~ismember(lower(string(cfg.Model.LossType)), ["hydro_signal_huber","hydrosignal_huber"])
    error('spatial2d:UnsupportedLoss','This release implements the HydroSignal Huber loss.');
end
lossLayer = HydrologySignalHuberRegressionLayer( ...
    cfg.Model.DefaultHuberDelta, cfg.Model.BiasPenaltyWeight, ...
    cfg.Model.CorrelationPenaltyWeight, cfg.Model.StdPenaltyWeight, ...
    cfg.Model.DifferencePenaltyWeight, cfg.Model.AnnualDifferencePenaltyWeight, 'huber_loss');

if cfg.Model.UseSpatioTemporalAttention
    mergeBranch = [
        mergeBranch
        SpatioTemporalAttention('st_attention')
        fullyConnectedLayer(1, Name='regression_fc')
        lossLayer
        ];
else
    mergeBranch = [
        mergeBranch
        fullyConnectedLayer(1, Name='regression_fc')
        lossLayer
        ];
end

lgraph = layerGraph(inputLayer);
lgraph = addLayers(lgraph, foldLayer);
lgraph = addLayers(lgraph, noiseLayer);
lgraph = addLayers(lgraph, cnnBranch);
lgraph = addLayers(lgraph, lstmBranch);
lgraph = addLayers(lgraph, mergeBranch);

lgraph = connectLayers(lgraph, 'input', 'fold/in');
lgraph = connectLayers(lgraph, 'fold/out', 'input_noise');
lgraph = connectLayers(lgraph, 'input_noise', 'cnn_conv1');
lgraph = connectLayers(lgraph, 'input_noise', 'center_pixel');
lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold_cnn/miniBatchSize');
lgraph = connectLayers(lgraph, 'fold/miniBatchSize', 'unfold_lstm/miniBatchSize');
lgraph = connectLayers(lgraph, 'cnn_fc', 'concat/in1');
lgraph = connectLayers(lgraph, 'lstm_fc', 'concat/in2');
end

