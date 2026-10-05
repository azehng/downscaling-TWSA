function [meanPred, stdPred] = qtp_mc_predict(net, source, numSamples, batchSize)
% MC inference on spatial-patch providers or explicit H-W-C-T sequence cells.
% Example: M=load(modelFile); [meanNorm,stdNorm]=qtp_mc_predict(M.net,X,20,64);
% X is an N-by-1 cell of normalized H-W-C-T sequences or QTPSpatialPatchSource.
% Return values have shape 1-T-N and remain in normalized anomaly units.
% Reverse M.Ps.Output to obtain anomalies; std uses division by Ps.Output.gain.
% Training climatology must be restored separately to obtain raw TWSA.
validateattributes(numSamples, {'numeric'}, {'scalar','integer','positive'});
validateattributes(batchSize, {'numeric'}, {'scalar','integer','positive'});
if isa(source,'QTPSpatialPatchSource')
    n = source.NumObservations;
    seqLen = source.SequenceLength;
elseif iscell(source) && ~isempty(source)
    n = numel(source);
    seqLen = size(source{1},4);
else
    error('spatial2d:WrongPredictionInput', ...
        'CNN2D requires spatial image-sequence cells or QTPSpatialPatchSource, not C-T-N tensors.');
end
meanPred = zeros(1,seqLen,n,'single');
m2 = zeros(1,seqLen,n,'single');
for pass=1:numSamples
    for first=1:batchSize:n
        idx = first:min(first+batchSize-1,n);
        if isa(source,'QTPSpatialPatchSource')
            cells = source.readIndices(idx);
        else
            cells = source(idx);
        end
        raw = predict(net,cells(:),'MiniBatchSize',batchSize);
        pred = zeros(1,seqLen,numel(idx),'single');
        if ~iscell(raw)
            error('spatial2d:PredictionShape', 'Expected sequence-to-sequence cell predictions.');
        end
        for k=1:numel(idx)
            assert(isequal(size(raw{k}),[1 seqLen]), 'Prediction must have one output per month.');
            pred(:,:,k) = single(raw{k});
        end
        delta = pred-meanPred(:,:,idx);
        meanPred(:,:,idx) = meanPred(:,:,idx)+delta/pass;
        m2(:,:,idx) = m2(:,:,idx)+delta.*(pred-meanPred(:,:,idx));
    end
end
stdPred = sqrt(max(m2/numSamples,0));
end
