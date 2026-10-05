classdef GaussianNoiseLayer < nnet.layer.Layer
    properties
        StdDev (1,1) double {mustBeNonnegative} = 0.05
    end

    methods
        function layer = GaussianNoiseLayer(stdDev, name)
            if nargin >= 1
                layer.StdDev = stdDev;
            end
            if nargin >= 2
                layer.Name = name;
            end
            layer.Description = "Gaussian noise regularization";
        end

        function Z = predict(~, X)
            Z = X;
        end

        function Z = forward(layer, X)
            Z = X + layer.StdDev .* randn(size(X), 'like', X);
        end
    end
end
