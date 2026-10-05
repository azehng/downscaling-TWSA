classdef SpatialChannelAttention2DLayer < nnet.layer.Layer
    methods
        function layer = SpatialChannelAttention2DLayer(name)
            layer.Name = name;
            layer.Description = "Channel and spatial attention on folded 2D frames";
        end
        function Z = predict(~, X)
            channelGate = 1 ./ (1 + exp(-mean(mean(X,1),2)));
            spatialGate = 1 ./ (1 + exp(-mean(X,3)));
            Z = X .* channelGate .* spatialGate;
        end
        function Z = forward(layer, X)
            Z = predict(layer, X);
        end
    end
end
