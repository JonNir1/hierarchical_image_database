%% shine_equalize.m
%
% Applies SHINE_color luminance/color equalization to all .png images
% under INPUT_DIR, preserving the subdirectory structure in OUTPUT_DIR.
%
% Because SHINE_color requires a flat input directory, the pipeline:
%   1. Flattens INPUT_DIR into a temporary flat directory, encoding
%      subdirectory paths into filenames using '__' as a separator.
%   2. Runs SHINE_color on the flat directory.
%   3. Restores the original subdirectory structure in OUTPUT_DIR.
%   4. Removes the temporary directory.
%
% NOTE: The flatten step assumes that no two source images share both the
% same filename AND the same encoded path prefix (i.e. filenames are
% globally unique within INPUT_DIR, or are unique when path-encoded).
% A warning is printed if a conflict is detected.
%
% Edit INPUT_DIR, OUTPUT_DIR, and SHINE_MODE before running.

%% --- Constants -----------------------------------------------------------
SCRIPT_DIR    = fileparts(mfilename('fullpath'));
repo_name     = 'hierarchical_image_database';
idx           = strfind(SCRIPT_DIR, repo_name);
ROOT_DIR      = SCRIPT_DIR(1 : idx + numel(repo_name) - 1);

INPUT_DIR     = fullfile(ROOT_DIR, 'images', 'pre_shine');
OUTPUT_DIR    = fullfile(ROOT_DIR, 'images', 'post_shine');
SHINE_DIR     = fullfile(ROOT_DIR, 'image_processing', 'SHINE_color_toolbox', ...
                         'SHINE_color_toolbox_v0.2');
SHINE_MODE    = 'LumEquated_histMatch';   % change to any SHINE output mode
TEMP_FLAT_DIR = fullfile(ROOT_DIR, 'images', '_shine_tmp_flat');

%% --- Collect all .png files recursively ---------------------------------
listing = dir(fullfile(INPUT_DIR, '**', '*.png'));
n = numel(listing);
fprintf('Found %d .png files under %s\n', n, INPUT_DIR);

if n == 0
    error('No .png files found in INPUT_DIR: %s', INPUT_DIR);
end

if ~exist(TEMP_FLAT_DIR, 'dir')
    mkdir(TEMP_FLAT_DIR);
end

%% --- Step 1: Flatten and collect alpha templates ------------------------
flat_names    = cell(n, 1);   % encoded flat filename (no extension)
rel_subpaths  = cell(n, 1);   % relative path within INPUT_DIR (for restore)
alpha_templates = cell(n, 1);

for i = 1:n
    src_path = fullfile(listing(i).folder, listing(i).name);

    % Encode the relative subdir into the filename
    rel_dir  = strrep(listing(i).folder, INPUT_DIR, '');
    rel_dir  = strtrim(rel_dir);
    % Remove leading separator if present
    if ~isempty(rel_dir) && (rel_dir(1) == filesep || rel_dir(1) == '/' || rel_dir(1) == '\')
        rel_dir = rel_dir(2:end);
    end

    if isempty(rel_dir)
        encoded_name = listing(i).name;
    else
        % Replace path separators with '__'
        encoded_rel = strrep(rel_dir, filesep, '__');
        encoded_rel = strrep(encoded_rel, '/', '__');
        encoded_rel = strrep(encoded_rel, '\', '__');
        [~, base, ext] = fileparts(listing(i).name);
        encoded_name = [encoded_rel '__' base ext];
    end

    flat_path = fullfile(TEMP_FLAT_DIR, encoded_name);

    % Conflict check
    if exist(flat_path, 'file')
        warning('Filename conflict — skipping duplicate: %s', src_path);
        continue;
    end

    copyfile(src_path, flat_path);

    flat_names{i}   = encoded_name;
    rel_subpaths{i} = fullfile(rel_dir, listing(i).name);  % relative path for restore

    % Read alpha channel for SHINE template
    [tmp_img, ~, alpha] = imread(flat_path);
    if isempty(alpha)
        alpha = 255 * ones(size(tmp_img, 1), size(tmp_img, 2), 'uint8');
        warning('No alpha channel in %s — using fully opaque mask.', src_path);
    end
    alpha_templates{i} = alpha;
end

% Remove any empty entries caused by skipped conflicts
valid = ~cellfun(@isempty, flat_names);
flat_names      = flat_names(valid);
rel_subpaths    = rel_subpaths(valid);
alpha_templates = alpha_templates(valid);

%% --- Step 2: Run SHINE_color --------------------------------------------
addpath(SHINE_DIR);
fprintf('Running SHINE_color (mode: %s) on %d images...\n', SHINE_MODE, numel(flat_names));

SHINE_color({}, alpha_templates, TEMP_FLAT_DIR, TEMP_FLAT_DIR);

%% --- Step 3: Restore subdirectory structure -----------------------------
shine_out_dir = fullfile(TEMP_FLAT_DIR, SHINE_MODE);

if ~exist(shine_out_dir, 'dir')
    % SHINE may use a slightly different folder name; try to find it
    d = dir(TEMP_FLAT_DIR);
    subdirs = {d([d.isdir] & ~strcmp({d.name},'.') & ~strcmp({d.name},'..')).name};
    if numel(subdirs) == 1
        shine_out_dir = fullfile(TEMP_FLAT_DIR, subdirs{1});
        fprintf('SHINE output found in: %s\n', shine_out_dir);
    else
        error('Cannot locate SHINE output subfolder in %s. Subdirs found: %s', ...
              TEMP_FLAT_DIR, strjoin(subdirs, ', '));
    end
end

for i = 1:numel(flat_names)
    shine_file = fullfile(shine_out_dir, flat_names{i});
    if ~exist(shine_file, 'file')
        warning('SHINE output not found for: %s', flat_names{i});
        continue;
    end

    dst_path = fullfile(OUTPUT_DIR, rel_subpaths{i});
    dst_dir  = fileparts(dst_path);
    if ~exist(dst_dir, 'dir')
        mkdir(dst_dir);
    end

    movefile(shine_file, dst_path);
end

fprintf('Output written to: %s\n', OUTPUT_DIR);

%% --- Step 4: Cleanup temp directory -------------------------------------
rmdir(TEMP_FLAT_DIR, 's');
fprintf('Temporary directory removed.\n');
