classdef QTPSpatialPatchSource < handle
    % File-backed predictor frames; only mini-batch spatial patches are expanded.
    properties (SetAccess=private)
        Meta
        FrameStoreDir
        Prefix
        InputPs
        FillValues
        PatchSize
        NumFeatures
        SequenceLength
        NumObservations
        FeatureNames
        Reference
        Cache
        CacheOrder = []
        MaxCachedYears
    end
    methods
        function obj = QTPSpatialPatchSource(meta, frameDir, prefix, inputPs, fillValues, cfg, featureNames, reference)
            obj.Meta = meta;
            obj.FrameStoreDir = frameDir;
            obj.Prefix = prefix;
            obj.InputPs = inputPs;
            obj.FillValues = single(fillValues(:));
            obj.PatchSize = cfg.Spatial.PatchSize;
            obj.NumFeatures = cfg.Dataset.NumFeatures;
            obj.SequenceLength = cfg.Dataset.SequenceLength;
            obj.NumObservations = numel(meta.Year);
            obj.FeatureNames = string(featureNames);
            obj.Reference = reference;
            obj.MaxCachedYears = cfg.Spatial.MaxCachedYears;
            obj.Cache = containers.Map('KeyType','double','ValueType','any');
            assert(numel(obj.FillValues)==obj.NumFeatures, 'Fill values must match channels.');
            for year = unique(meta.Year(:))'
                file = obj.frameFile(year);
                if ~isfile(file)
                    error('spatial2d:MissingFrames', 'Missing prepared spatial predictor frames: %s. Configure cfg.Spatial.FrameStoreDir.', file);
                end
            end
        end
        function cells = readIndices(obj, indices)
            indices = indices(:);
            cells = cell(numel(indices),1);
            years = obj.Meta.Year(indices);
            for year = unique(years(:))'
                cube = obj.loadYear(year);
                positions = find(years==year);
                for k = positions'
                    idx = indices(k);
                    cells{k} = qtp_extract_spatial_patch(cube, obj.Meta.Row(idx), ...
                        obj.Meta.Col(idx), obj.PatchSize, obj.FillValues);
                end
            end
        end
        function cube = loadYear(obj, year)
            if isKey(obj.Cache, year)
                cube = obj.Cache(year);
                return;
            end
            S = load(obj.frameFile(year), 'FeatureCube','FeatureNames','Reference');
            if ~isequal(string(S.FeatureNames), obj.FeatureNames) || ...
                    size(S.FeatureCube,3)~=obj.NumFeatures || size(S.FeatureCube,4)~=obj.SequenceLength
                error('spatial2d:FrameSchemaMismatch', 'Frame channels or sequence length differ from the dataset.');
            end
            if ~isequal(S.Reference, obj.Reference)
                error('spatial2d:ReferenceMismatch', 'Frame raster reference differs from sample metadata.');
            end
            shape = size(S.FeatureCube);
            flat = reshape(permute(S.FeatureCube,[3 1 2 4]), obj.NumFeatures, []);
            flat = single(mapminmax('apply', double(flat), obj.InputPs));
            cube = permute(reshape(flat, [obj.NumFeatures, shape(1),shape(2),obj.SequenceLength]), [2 3 1 4]);
            if numel(obj.CacheOrder)>=obj.MaxCachedYears
                remove(obj.Cache,obj.CacheOrder(1));
                obj.CacheOrder(1) = [];
            end
            obj.Cache(year) = cube;
            obj.CacheOrder(end+1) = year;
        end
        function file = frameFile(obj, year)
            file = fullfile(obj.FrameStoreDir, sprintf('%s_%04d.mat',obj.Prefix,year));
        end
    end
end
