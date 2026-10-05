classdef MonteCarloDropoutLayer < nnet.layer.Layer
    properties
        Probability (1,1) double {mustBeGreaterThanOrEqual(Probability, 0), mustBeLessThan(Probability, 1)} = 0.2
    end

    methods
        function layer = MonteCarloDropoutLayer(probability, name)
            if nargin >= 1
                layer.Probability = probability;
            end
            if nargin >= 2
                layer.Name = name;
            end
            layer.Description = "Dropout enabled in training and prediction";
        end

        function Z = predict(layer, X)
            Z = iApplyDropout(layer, X);
        end

        function Z = forward(layer, X)
            Z = iApplyDropout(layer, X);
        end
    end
end

function Z = iApplyDropout(layer, X)
keepProbability = 1 - layer.Probability;
mask = rand(size(X), 'like', X) < keepProbability;
Z = X .* mask ./ keepProbability;
end
