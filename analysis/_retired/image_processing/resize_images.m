%% resize_images.m
%
% Recursively collects all .png files under INPUT_DIR, pads non-square
% images with white (RGB=255) and transparent (alpha=0) borders, resizes
% to TARGET_SIZE x TARGET_SIZE, and writes to OUTPUT_DIR preserving the
% input subdirectory structure.
%
% Source images must have an alpha channel (4-channel RGBA PNG).
% Uses gray2rgb.m from image_processing/helpers/ for any single-channel inputs.
%
% Edit INPUT_DIR and OUTPUT_DIR before running.

%% --- Constants -----------------------------------------------------------
SCRIPT_DIR  = fileparts(mfilename('fullpath'));
addpath(fullfile(SCRIPT_DIR, 'helpers'));
repo_name   = 'hierarchical_image_database';
idx         = strfind(SCRIPT_DIR, repo_name);
ROOT_DIR    = SCRIPT_DIR(1 : idx + numel(repo_name) - 1);

INPUT_DIR   = fullfile(ROOT_DIR, 'source_datasets', '');  % <-- fill in subpath
OUTPUT_DIR  = fullfile(ROOT_DIR, 'images', 'pre_shine');
TARGET_SIZE = 175;

%% --- Collect all .png files recursively ---------------------------------
listing = dir(fullfile(INPUT_DIR, '**', '*.png'));
fprintf('Found %d .png files under %s\n', numel(listing), INPUT_DIR);

%% --- Process each image -------------------------------------------------
for i = 1:numel(listing)
    src_path = fullfile(listing(i).folder, listing(i).name);

    % Reconstruct destination path, mirroring subdir structure
    rel_path = strrep(listing(i).folder, INPUT_DIR, '');
    dst_dir  = fullfile(OUTPUT_DIR, rel_path);
    dst_path = fullfile(dst_dir, listing(i).name);

    if ~exist(dst_dir, 'dir')
        mkdir(dst_dir);
    end

    % Read image + alpha
    [I, ~, alpha] = imread(src_path);

    % Handle grayscale (single-channel) inputs
    if size(I, 3) == 1
        I = gray2rgb(I);
    end

    % Handle missing alpha channel
    if isempty(alpha)
        alpha = 255 * ones(size(I, 1), size(I, 2), 'uint8');
        warning('No alpha channel in %s — using fully opaque mask.', src_path);
    end

    [h, w, ~] = size(I);

    % Skip if already the right size
    if h == TARGET_SIZE && w == TARGET_SIZE
        imwrite(I, dst_path, 'Alpha', alpha);
        continue;
    end

    % Pad non-square images to a square canvas
    if h ~= w
        side   = max(h, w);
        top    = ceil((side - h) / 2);
        left   = ceil((side - w) / 2);

        newI     = 255 * ones(side, side, 3, 'uint8');   % white background
        newAlpha = zeros(side, side, 'uint8');            % transparent background

        newI(top+1:top+h, left+1:left+w, :) = I;
        newAlpha(top+1:top+h, left+1:left+w) = alpha;

        I     = newI;
        alpha = newAlpha;
    end

    % Resize
    I     = imresize(I,     [TARGET_SIZE, TARGET_SIZE]);
    alpha = imresize(alpha, [TARGET_SIZE, TARGET_SIZE]);

    imwrite(I, dst_path, 'Alpha', alpha);
    fprintf('[%d/%d] %s\n', i, numel(listing), listing(i).name);
end

fprintf('Done. Output written to: %s\n', OUTPUT_DIR);
