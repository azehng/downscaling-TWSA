function patch = qtp_extract_spatial_patch(cube, row, col, patchSize, fillValues)
% Cube is normalized H-by-W-by-C-by-T. Padding replicates raster edges.
validateattributes(patchSize, {'numeric'}, {'numel',2,'positive','integer'});
if any(mod(patchSize,2) ~= 1)
    error('spatial2d:EvenPatch', 'Patch dimensions must be odd.');
end
validateattributes(row, {'numeric'}, {'scalar','integer','>=',1,'<=',size(cube,1)});
validateattributes(col, {'numeric'}, {'scalar','integer','>=',1,'<=',size(cube,2)});
dr = -(patchSize(1)-1)/2:(patchSize(1)-1)/2;
dc = -(patchSize(2)-1)/2:(patchSize(2)-1)/2;
rr = max(1, min(size(cube,1), row+dr));
cc = max(1, min(size(cube,2), col+dc));
patch = cube(rr,cc,:,:);
for channel = 1:size(cube,3)
    block = patch(:,:,channel,:);
    block(~isfinite(block)) = fillValues(channel);
    patch(:,:,channel,:) = block;
end
end
