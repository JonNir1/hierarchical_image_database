function check_output_dir(output_dir)
% check_output_dir  Guard OUTPUT_DIR before a SHINE equalization run.
%
%   check_output_dir(output_dir)
%
%   * If OUTPUT_DIR does not exist, it is created.
%   * If OUTPUT_DIR exists but is empty, the call returns silently.
%   * If OUTPUT_DIR contains any .png files (recursively), an error is
%     thrown with identifier 'shine_equalize:outputDirNotEmpty' so that
%     existing results are never silently overwritten.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
    return;
end

existing = dir(fullfile(output_dir, '**', '*.png'));
if ~isempty(existing)
    error('shine_equalize:outputDirNotEmpty', ...
          ['OUTPUT_DIR already contains %d .png file(s) -- aborting to ' ...
           'prevent overwrite.\nClear or rename it before re-running:\n  %s'], ...
          numel(existing), output_dir);
end
end
