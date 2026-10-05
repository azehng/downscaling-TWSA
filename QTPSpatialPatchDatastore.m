classdef QTPSpatialPatchDatastore < matlab.io.Datastore & ...
        matlab.io.datastore.MiniBatchable & matlab.io.datastore.Shuffleable
    properties
        MiniBatchSize = 256
    end
    properties (SetAccess=protected)
        NumObservations
    end
    properties (SetAccess=private)
        Source
        Targets
        SourceIndices
        Order
        Cursor = 1
    end
    methods
        function ds = QTPSpatialPatchDatastore(source, targets, indices, batchSize)
            ds.Source = source;
            ds.Targets = targets;
            ds.SourceIndices = indices(:);
            ds.NumObservations = numel(indices);
            ds.Order = (1:ds.NumObservations)';
            ds.MiniBatchSize = batchSize;
            assert(size(targets,3)==source.NumObservations, 'Targets and source observations must agree.');
        end
        function tf = hasdata(ds)
            tf = ds.Cursor<=ds.NumObservations;
        end
        function [data, info] = read(ds)
            if ~hasdata(ds)
                error('spatial2d:NoData', 'No unread observations. Call reset.');
            end
            last = min(ds.NumObservations, ds.Cursor+ds.MiniBatchSize-1);
            pos = ds.Order(ds.Cursor:last);
            idx = ds.SourceIndices(pos);
            inputs = ds.Source.readIndices(idx);
            targets = cell(numel(idx),1);
            for k=1:numel(idx)
                targets{k} = ds.Targets(:,:,idx(k));
            end
            data = table(inputs, targets, 'VariableNames',{'Input','Response'});
            info = struct('SourceIndices',idx);
            ds.Cursor = last+1;
        end
        function reset(ds)
            ds.Cursor = 1;
        end
        function fraction = progress(ds)
            fraction = min(1,(ds.Cursor-1)/max(ds.NumObservations,1));
        end
        function shuffled = shuffle(ds)
            shuffled = QTPSpatialPatchDatastore(ds.Source,ds.Targets,ds.SourceIndices,ds.MiniBatchSize);
            shuffled.Order = ds.Order(randperm(ds.NumObservations));
        end
    end
end
