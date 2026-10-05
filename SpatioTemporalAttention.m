classdef SpatioTemporalAttention < nnet.layer.Layer
    properties (Learnable)
        ChannelScale
        TemporalScale
        Alpha
    end

    methods
        function layer = SpatioTemporalAttention(name)
            if nargin >= 1
                layer.Name = name;
            end
            layer.Description = "Learnable spatiotemporal attention";
        end

        function layer = initialize(layer, X)
            [numChannels, sequenceLength] = iInferShape(X);
            layer.ChannelScale = dlarray(ones(numChannels, 1, 'single'));
            layer.TemporalScale = dlarray(ones(1, sequenceLength, 'single'));
            layer.Alpha = dlarray(single(0.1));
        end

        function Z = predict(layer, X)
            [X3, undo] = iToCST(X);

            channelDescriptor = mean(X3, 2);
            temporalDescriptor = mean(X3, 1);

            channelGate = sigmoid(channelDescriptor .* layer.ChannelScale);
            channelGate = reshape(channelGate, size(channelGate, 1), 1, size(channelGate, 3));

            temporalGate = sigmoid(temporalDescriptor .* layer.TemporalScale);
            temporalGate = reshape(temporalGate, 1, size(temporalGate, 2), size(temporalGate, 3));

            attention = channelGate .* temporalGate;
            Z = undo(X3 + layer.Alpha .* (X3 .* attention));
        end

        function Z = forward(layer, X)
            Z = predict(layer, X);
        end
    end
end

function [numChannels, sequenceLength] = iInferShape(X)
sz = size(X);
switch ndims(X)
    case 2
        numChannels = sz(1);
        sequenceLength = sz(2);
    case 3
        numChannels = sz(1);
        sequenceLength = sz(2);
    case 4
        numChannels = sz(3);
        sequenceLength = sz(2);
    otherwise
        error('SpatioTemporalAttention:UnsupportedInput', 'Unsupported input rank: %d', ndims(X));
end
end

function [X3, undo] = iToCST(X)
sz = size(X);
switch ndims(X)
    case 2
        X3 = reshape(X, sz(1), sz(2), 1);
        undo = @(Y) reshape(Y, sz);
    case 3
        X3 = X;
        undo = @(Y) Y;
    case 4
        X3 = permute(X, [3 2 4 1]);
        X3 = reshape(X3, sz(3), sz(2), sz(4) * sz(1));
        undo = @(Y) ipermute(reshape(Y, sz(3), sz(2), sz(4), sz(1)), [3 2 4 1]);
    otherwise
        error('SpatioTemporalAttention:UnsupportedInput', 'Unsupported input rank: %d', ndims(X));
end
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
