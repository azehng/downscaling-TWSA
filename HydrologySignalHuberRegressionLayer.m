classdef HydrologySignalHuberRegressionLayer < nnet.layer.RegressionLayer
    properties
        Delta (1,1) double {mustBePositive} = 1.0
        BiasPenaltyWeight (1,1) double {mustBeNonnegative} = 0.01
        CorrelationPenaltyWeight (1,1) double {mustBeNonnegative} = 0.10
        StdPenaltyWeight (1,1) double {mustBeNonnegative} = 0.10
        DifferencePenaltyWeight (1,1) double {mustBeNonnegative} = 0.05
        AnnualDifferencePenaltyWeight (1,1) double {mustBeNonnegative} = 0.00
        Epsilon (1,1) double {mustBePositive} = 1e-6
    end

    methods
        function layer = HydrologySignalHuberRegressionLayer(delta, biasWeight, corrWeight, stdWeight, diffWeight, annualDiffWeight, name)
            if nargin >= 1 && ~isempty(delta)
                layer.Delta = delta;
            end
            if nargin >= 2 && ~isempty(biasWeight)
                layer.BiasPenaltyWeight = biasWeight;
            end
            if nargin >= 3 && ~isempty(corrWeight)
                layer.CorrelationPenaltyWeight = corrWeight;
            end
            if nargin >= 4 && ~isempty(stdWeight)
                layer.StdPenaltyWeight = stdWeight;
            end
            if nargin >= 5 && ~isempty(diffWeight)
                layer.DifferencePenaltyWeight = diffWeight;
            end
            if nargin >= 6 && ~isempty(annualDiffWeight)
                layer.AnnualDifferencePenaltyWeight = annualDiffWeight;
            end
            if nargin >= 7
                layer.Name = name;
            end
            layer.Description = "Huber loss with bias, correlation, amplitude, monthly-change, and annual-change penalties";
        end

        function loss = forwardLoss(layer, Y, T)
            err = Y - T;
            absErr = abs(err);
            quadraticMask = absErr <= layer.Delta;
            huberLoss = 0.5 .* (err .^ 2) .* quadraticMask + ...
                (~quadraticMask) .* (layer.Delta .* absErr - 0.5 * layer.Delta^2);

            batchSize = size(Y, ndims(Y));
            baseLoss = sum(huberLoss(:)) ./ max(batchSize, 1);

            yFlat = reshape(Y, [], batchSize);
            tFlat = reshape(T, [], batchSize);
            errFlat = yFlat - tFlat;

            sampleBias = mean(errFlat, 1);
            biasPenalty = mean(sampleBias .^ 2);

            yCentered = yFlat - mean(yFlat, 1);
            tCentered = tFlat - mean(tFlat, 1);
            yStd = sqrt(mean(yCentered .^ 2, 1) + layer.Epsilon);
            tStd = sqrt(mean(tCentered .^ 2, 1) + layer.Epsilon);

            numerator = sum(yCentered .* tCentered, 1);
            denominator = sqrt(sum(yCentered .^ 2, 1) .* sum(tCentered .^ 2, 1) + layer.Epsilon);
            corrValue = numerator ./ denominator;
            corrPenalty = mean(1 - corrValue);

            stdRatio = yStd ./ (tStd + layer.Epsilon);
            stdPenalty = mean((stdRatio - 1) .^ 2);

            if size(yFlat, 1) >= 2
                yDiff = yFlat(2:end, :) - yFlat(1:end-1, :);
                tDiff = tFlat(2:end, :) - tFlat(1:end-1, :);
                diffErr = (yDiff - tDiff) .^ 2;
                diffPenalty = mean(diffErr(:));
            else
                diffPenalty = 0;
            end

            if size(yFlat, 1) >= 13
                yAnnualDiff = yFlat(13:end, :) - yFlat(1:end-12, :);
                tAnnualDiff = tFlat(13:end, :) - tFlat(1:end-12, :);
                annualDiffErr = (yAnnualDiff - tAnnualDiff) .^ 2;
                annualDiffPenalty = mean(annualDiffErr(:));
            else
                annualDiffPenalty = 0;
            end

            loss = baseLoss + ...
                layer.BiasPenaltyWeight .* biasPenalty + ...
                layer.CorrelationPenaltyWeight .* corrPenalty + ...
                layer.StdPenaltyWeight .* stdPenalty + ...
                layer.DifferencePenaltyWeight .* diffPenalty + ...
                layer.AnnualDifferencePenaltyWeight .* annualDiffPenalty;
        end
    end
end
