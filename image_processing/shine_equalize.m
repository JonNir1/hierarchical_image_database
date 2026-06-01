%% shine_equalize.m
%
% Applies the SHINE_color toolbox (Willenbockel et al., 2010; color
% adaptation by Dal Ben, 2019) to all .png images under INPUT_DIR,
% preserving the subdirectory structure in OUTPUT_DIR.
%
% Target configuration: LumEquated_histMatch
%   = luminance equating via histogram matching (SHINE mode 2, histMatch),
%     with figure-ground separation driven by the per-image alpha channel
%     (wholeIm = 3), template background luminance = 0 (transparent).
%
% IMPORTANT: SHINE_color is a vendored, peer-reviewed toolbox and is NOT
% modified. Because its input() prompts cannot be driven from outside, this
% wrapper prints an exact answer sheet before handing control to SHINE_color
% -- type the listed values, in order, at each command-window prompt.
%
% Pipeline:
%   1. Flattens INPUT_DIR into TEMP_FLAT_IN, encoding subdirectory paths
%      into filenames using '__' as a separator.
%   2. Reads alpha templates in the same alphabetical order SHINE_color uses
%      (via readImages.m), so templ{im} aligns positionally with each image.
%   3. Runs SHINE_color, writing results to TEMP_FLAT_OUT.
%   4. Restores the original subdirectory structure in OUTPUT_DIR using a
%      flat_name -> relative_subpath map built during flattening.
%   5. Removes both temporary directories.
%
% NOTE: The flatten step assumes encoded filenames are globally unique.
% A warning is printed and the duplicate skipped if a conflict is detected.
%
% REQUIRES: Image Processing Toolbox (for medfilt2, called by SHINE_color's
%   separate.m during figure-ground segmentation).
%   Install via: Home -> Add-Ons -> Get Add-Ons -> "Image Processing Toolbox".
%
% Edit INPUT_DIR / OUTPUT_DIR before running.

%% --- Constants -----------------------------------------------------------
SCRIPT_DIR    = fileparts(mfilename('fullpath'));
repo_name     = 'hierarchical_image_database';
idx           = strfind(SCRIPT_DIR, repo_name);
ROOT_DIR      = SCRIPT_DIR(1 : idx + numel(repo_name) - 1);

INPUT_DIR     = fullfile(ROOT_DIR, 'images', 'pre_shine');
OUTPUT_DIR    = fullfile(ROOT_DIR, 'images', 'post_shine');
SHINE_DIR     = fullfile(SCRIPT_DIR, 'SHINE_color_toolbox');
HELPERS_DIR   = fullfile(SCRIPT_DIR, 'helpers');
TEMP_FLAT_IN  = fullfile(ROOT_DIR, 'images', '_shine_tmp_in');
TEMP_FLAT_OUT = fullfile(ROOT_DIR, 'images', '_shine_tmp_out');
addpath(HELPERS_DIR);

%% --- Collect all .png files recursively ---------------------------------
listing = dir(fullfile(INPUT_DIR, '**', '*.png'));
n = numel(listing);
fprintf('Found %d .png files under %s\n', n, INPUT_DIR);

if n == 0
    error('No .png files found in INPUT_DIR: %s', INPUT_DIR);
end

% Guard OUTPUT_DIR: create if absent, error if non-empty
check_output_dir(OUTPUT_DIR);

% Fresh temp dirs (wipe any leftover from a previous interrupted run)
for d = {TEMP_FLAT_IN, TEMP_FLAT_OUT}
    if exist(d{1}, 'dir'); rmdir(d{1}, 's'); end
    mkdir(d{1});
end

%% --- Step 1: Flatten INPUT_DIR into TEMP_FLAT_IN -------------------------
% Record a flat_name -> relative_subpath map for the restore step.
name2rel = containers.Map('KeyType', 'char', 'ValueType', 'char');

for i = 1:n
    src_path = fullfile(listing(i).folder, listing(i).name);

    % Encode the relative subdir into the filename
    rel_dir  = strrep(listing(i).folder, INPUT_DIR, '');
    rel_dir  = strtrim(rel_dir);
    % Remove leading separator if present
    if ~isempty(rel_dir) && (rel_dir(1) == filesep || rel_dir(1) == '/' || rel_dir(1) == '\')
        rel_dir = rel_dir(2:end);
    end

    encoded_name = encode_flat_name(rel_dir, listing(i).name);

    flat_path = fullfile(TEMP_FLAT_IN, encoded_name);

    % Conflict check
    if isKey(name2rel, encoded_name) || exist(flat_path, 'file')
        warning('Filename conflict -- skipping duplicate: %s', src_path);
        continue;
    end

    copyfile(src_path, flat_path);
    name2rel(encoded_name) = fullfile(rel_dir, listing(i).name);
end

%% --- Step 2: Read alpha templates in SHINE/readImages order -------------
% readImages.m loads files via dir(fullfile(pathname,'*.png')) -- alphabetical
% -- and SHINE_color writes each result back positionally (templ{im}).
% We must build alpha_templates in that exact order so masks align with images.
flat_listing = dir(fullfile(TEMP_FLAT_IN, '*.png'));
m = numel(flat_listing);
alpha_templates = cell(m, 1);

for im = 1:m
    [img, ~, alpha] = imread(fullfile(TEMP_FLAT_IN, flat_listing(im).name));
    if isempty(alpha)
        alpha = 255 * ones(size(img, 1), size(img, 2), 'uint8');
        warning('No alpha channel in %s -- using fully opaque mask.', ...
                flat_listing(im).name);
    end
    alpha_templates{im} = alpha;
end

%% --- Step 3: Run SHINE_color (interactive) ------------------------------
addpath(SHINE_DIR);

fprintf('\n');
fprintf('========================================================================\n');
fprintf(' SHINE_color is interactive. Enter the following at each prompt, in order\n');
fprintf(' (target config: LumEquated_histMatch + figure-ground via alpha mask):\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('   1.  Input     [1=images, 2=video]                          ->  1\n');
fprintf('   2.  Type the image format  [e.g., jpg, png]                ->  png\n');
fprintf('   3.  SHINE_color options    [1=default, 2=custom]           ->  2\n');
fprintf('   4.  Matching mode [1=luminance, 2=spatial freq, 3=both]    ->  1\n');
fprintf('   5.  Luminance option [1=lumMatch, 2=histMatch]             ->  2\n');
fprintf('   6.  Optimize SSIM    [1=no, 2=yes]                         ->  1\n');
fprintf('   7.  Templ background [1=specify lum, 2=find automatically] ->  1\n');
fprintf('   8.  Enter lum value  [integer between 0 and 255]           ->  0\n');
fprintf('------------------------------------------------------------------------\n');
fprintf(' Running on %d images. Output -> %s\n', m, TEMP_FLAT_OUT);
fprintf('========================================================================\n\n');

SHINE_color({}, alpha_templates, TEMP_FLAT_IN, TEMP_FLAT_OUT);

%% --- Step 4: Restore subdirectory structure -----------------------------
out_listing = dir(fullfile(TEMP_FLAT_OUT, '*.png'));
if isempty(out_listing)
    error(['SHINE_color produced no .png output in %s. ' ...
           'Did the run complete (rather than being quit at a prompt)?'], ...
          TEMP_FLAT_OUT);
end

restored = 0;
for i = 1:numel(out_listing)
    encoded_name = out_listing(i).name;
    if ~isKey(name2rel, encoded_name)
        warning('Unexpected SHINE output with no source mapping: %s', encoded_name);
        continue;
    end

    dst_path = fullfile(OUTPUT_DIR, name2rel(encoded_name));
    dst_dir  = fileparts(dst_path);
    if ~exist(dst_dir, 'dir'); mkdir(dst_dir); end

    movefile(fullfile(TEMP_FLAT_OUT, encoded_name), dst_path);
    restored = restored + 1;
end

% Warn about any input that produced no SHINE output
out_names = {out_listing.name};
in_names  = name2rel.keys;
missing   = setdiff(in_names, out_names);
for k = 1:numel(missing)
    warning('No SHINE output found for input: %s', missing{k});
end

fprintf('Restored %d / %d images to: %s\n', restored, numel(in_names), OUTPUT_DIR);

%% --- Step 5: Cleanup temp directories -----------------------------------
rmdir(TEMP_FLAT_IN, 's');
rmdir(TEMP_FLAT_OUT, 's');
fprintf('Temporary directories removed.\n');
