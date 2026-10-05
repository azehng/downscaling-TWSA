classdef CenterPixel2DLayer < nnet.layer.Layer
    methods
        function layer = CenterPixel2DLayer(name)
            layer.Name = name;
            layer.Description = "Centre spatial pixel; preserve feature channels and folded frames";
        end
        function Z = predict(~, X)
            r = floor(size(X,1)/2) + 1;
            c = floor(size(X,2)/2) + 1;
            Z = X(r,c,:,:);
        end
        function Z = forward(layer, X)
            Z = predict(layer, X);
        end
    end
end
