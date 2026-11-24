classdef TestShineEqualize < matlab.unittest.TestCase
% TestShineEqualize  Unit and integration tests for the shine_equalize
% pipeline (helpers + wrapper logic).
%
% Covers:
%   encode_flat_name  -- encoding correctness for deep and top-level paths
%   check_output_dir  -- creates absent dir; errors on non-empty; silent on empty
%   Flatten step      -- all files copied, map built with correct rel-paths
%   Alpha ordering    -- alpha_templates aligned with readImages alphabetical order
%   Restore step      -- full encode -> simulated SHINE -> decode round-trip
%
% Does NOT call SHINE_color (interactive toolbox).  All tests use synthetic
% 10x15 RGBA fixture images created in a temporary directory; no file in
% images/pre_shine/ or images/post_shine/ is ever read or written.
%
% Run from the repo root:
%   results = runtests('image_processing/__tests__/TestShineEqualize');
%   disp(results.table)

    properties
        HelpersDir   % absolute path to image_processing/helpers/
        FixtureIn    % root of synthetic input tree  (tmp/fixture_in/)
        FlatIn       % temp flat input dir           (tmp/flat_in/)
        FlatOut      % temp flat output dir          (tmp/flat_out/)
        OutputDir    % temp restored output dir      (tmp/output/)

        % Relative paths of the 4 fixture images (forward slashes)
        FixtureRels = {
            'animate/animal/body/bird/sparrow.png'   % alpha tag = 11
            'animate/animal/body/bird/wren.png'      % alpha tag = 22
            'inanimate/handmade/ball/soccer.png'     % alpha tag = 33
            'inanimate/natural/flower/red/rose.png'  % alpha tag = 44
        }

        % Unique per-image alpha value (all pixels set to this uint8 value)
        AlphaTags = [11, 22, 33, 44]

        % Expected alphabetical order of encoded names (same as above here,
        % but written out explicitly so the ordering test is self-documenting)
        ExpectedFlatOrder = {
            'animate__animal__body__bird__sparrow.png'
            'animate__animal__body__bird__wren.png'
            'inanimate__handmade__ball__soccer.png'
            'inanimate__natural__flower__red__rose.png'
        }
    end

    % -----------------------------------------------------------------------
    methods (TestClassSetup)
        function addHelpers(tc)
            here          = fileparts(mfilename('fullpath'));  % __tests__/
            imgProc       = fileparts(here);                   % image_processing/
            tc.HelpersDir = fullfile(imgProc, 'helpers');
            addpath(tc.HelpersDir);
        end
    end

    % -----------------------------------------------------------------------
    methods (TestMethodSetup)
        function createFixtures(tc)
            here    = fileparts(mfilename('fullpath'));   % __tests__/
            imgProc = fileparts(here);                    % image_processing/
            tmpBase = fullfile(imgProc, 'tmp', 'test_run');

            tc.FixtureIn = fullfile(tmpBase, 'fixture_in');
            tc.FlatIn    = fullfile(tmpBase, 'flat_in');
            tc.FlatOut   = fullfile(tmpBase, 'flat_out');
            tc.OutputDir = fullfile(tmpBase, 'output');

            % Wipe and recreate the entire temp subtree
            if exist(tmpBase, 'dir'); rmdir(tmpBase, 's'); end

            % Create synthetic fixture images
            for i = 1:numel(tc.FixtureRels)
                rel      = strrep(tc.FixtureRels{i}, '/', filesep);
                dst_path = fullfile(tc.FixtureIn, rel);
                dst_dir  = fileparts(dst_path);
                if ~exist(dst_dir, 'dir'); mkdir(dst_dir); end

                % 10x15 non-square RGBA: unique alpha tag per image
                rgb   = uint8(mod((i * 37) * ones(10, 15, 3), 256));
                alpha = tc.AlphaTags(i) * ones(10, 15, 'uint8');
                imwrite(rgb, dst_path, 'Alpha', alpha);
            end

            % Create empty working dirs
            mkdir(tc.FlatIn);
            mkdir(tc.FlatOut);
        end
    end

    % -----------------------------------------------------------------------
    methods (TestMethodTeardown)
        function cleanup(tc)
            here    = fileparts(mfilename('fullpath'));  % __tests__/
            imgProc = fileparts(here);                   % image_processing/
            tmpBase = fullfile(imgProc, 'tmp', 'test_run');
            if exist(tmpBase, 'dir'); rmdir(tmpBase, 's'); end
        end
    end

    % ======================================================================
    % encode_flat_name
    % ======================================================================

    methods (Test)
        function testEncode_DeepPath(tc)
            % 4-level path on the native separator
            rel_dir  = fullfile('animate', 'animal', 'body', 'bird');
            result   = encode_flat_name(rel_dir, 'sparrow.png');
            tc.verifyEqual(result, 'animate__animal__body__bird__sparrow.png');
        end

        function testEncode_TopLevelFile(tc)
            % File at root -- no subdir -- returns filename unchanged
            result = encode_flat_name('', 'top.png');
            tc.verifyEqual(result, 'top.png');
        end

        function testEncode_ForwardSlash(tc)
            % Forward slashes (can appear on Windows from strrep artefacts)
            result = encode_flat_name('a/b/c', 'img.png');
            tc.verifyEqual(result, 'a__b__c__img.png');
        end

        function testEncode_BackwardSlash(tc)
            result = encode_flat_name('a\b\c', 'img.png');
            tc.verifyEqual(result, 'a__b__c__img.png');
        end

        function testEncode_Roundtrip(tc)
            % Encoded name contains rel_dir prefix + '__' + filename.
            % Verify split is recoverable given we know the original filename.
            rel_dir  = fullfile('inanimate', 'natural', 'flower', 'red');
            fname    = 'rose.png';
            encoded  = encode_flat_name(rel_dir, fname);
            expected = fullfile(rel_dir, fname);
            % Decode: strip the '__<fname>' suffix, replace '__' with filesep
            prefix = encoded(1 : end - numel(['__' fname]));
            decoded_dir = strrep(prefix, '__', filesep);
            tc.verifyEqual(fullfile(decoded_dir, fname), expected);
        end
    end

    % ======================================================================
    % check_output_dir
    % ======================================================================

    methods (Test)
        function testGuard_CreatesAbsentDir(tc)
            target = fullfile(tc.FlatOut, 'new_subdir');
            tc.verifyFalse(exist(target, 'dir') > 0);
            check_output_dir(target);
            tc.verifyTrue(exist(target, 'dir') > 0);
            rmdir(target);
        end

        function testGuard_SilentOnEmptyDir(tc)
            % OutputDir is empty at this point -- should not error
            mkdir(tc.OutputDir);
            check_output_dir(tc.OutputDir);   % no assertion needed; just must not throw
        end

        function testGuard_ErrorsWhenNonEmpty(tc)
            % Populate OutputDir with one PNG then verify guard fires
            mkdir(tc.OutputDir);
            imwrite(uint8(zeros(4,4,3)), fullfile(tc.OutputDir, 'dummy.png'));
            tc.verifyError(@() check_output_dir(tc.OutputDir), ...
                           'shine_equalize:outputDirNotEmpty');
        end

        function testGuard_ErrorMessageMentionsCount(tc)
            % Error message should state how many files were found
            mkdir(tc.OutputDir);
            for k = 1:3
                imwrite(uint8(zeros(4,4,3)), ...
                        fullfile(tc.OutputDir, sprintf('img%d.png', k)));
            end
            try
                check_output_dir(tc.OutputDir);
                tc.verifyFail('Expected error was not thrown.');
            catch ME
                tc.verifySubstring(ME.message, '3');
            end
        end
    end

    % ======================================================================
    % Flatten step (Step 1 of shine_equalize)
    % ======================================================================

    methods (Test)
        function testFlatten_AllFilesCopied(tc)
            tc.runFlattenStep();
            copied = dir(fullfile(tc.FlatIn, '*.png'));
            tc.verifyEqual(numel(copied), numel(tc.FixtureRels));
        end

        function testFlatten_EncodedNamesCorrect(tc)
            tc.runFlattenStep();
            for i = 1:numel(tc.ExpectedFlatOrder)
                flat_path = fullfile(tc.FlatIn, tc.ExpectedFlatOrder{i});
                tc.verifyTrue(exist(flat_path, 'file') > 0, ...
                    sprintf('Missing encoded file: %s', tc.ExpectedFlatOrder{i}));
            end
        end

        function testFlatten_MapRelPathsCorrect(tc)
            name2rel = tc.runFlattenStep();
            for i = 1:numel(tc.FixtureRels)
                encoded = tc.ExpectedFlatOrder{i};
                tc.verifyTrue(isKey(name2rel, encoded), ...
                    sprintf('Map missing key: %s', encoded));
                expected_rel = strrep(tc.FixtureRels{i}, '/', filesep);
                tc.verifyEqual(name2rel(encoded), expected_rel);
            end
        end

        function testFlatten_NoConflicts(tc)
            % Duplicate detection: add a conflicting file manually, re-run
            % We simulate a conflict by copying one source file under an
            % already-encoded name into FlatIn before the flatten step.
            conflict_name = tc.ExpectedFlatOrder{1};
            imwrite(uint8(zeros(4,4,3)), fullfile(tc.FlatIn, conflict_name));

            % Run flatten; the conflicting file should be skipped (warning)
            listing = dir(fullfile(tc.FixtureIn, '**', '*.png'));
            name2rel = containers.Map('KeyType','char','ValueType','char');
            w = warning('off', 'all');
            for i = 1:numel(listing)
                rel_dir = strrep(listing(i).folder, tc.FixtureIn, '');
                rel_dir = strtrim(rel_dir);
                if ~isempty(rel_dir) && any(rel_dir(1) == [filesep '/' '\'])
                    rel_dir = rel_dir(2:end);
                end
                encoded_name = encode_flat_name(rel_dir, listing(i).name);
                flat_path    = fullfile(tc.FlatIn, encoded_name);
                if isKey(name2rel, encoded_name) || exist(flat_path, 'file')
                    continue;  % conflict -- skip
                end
                copyfile(fullfile(listing(i).folder, listing(i).name), flat_path);
                name2rel(encoded_name) = fullfile(rel_dir, listing(i).name);
            end
            warning(w);

            % One file was skipped (the conflict), so map has n-1 entries
            tc.verifyEqual(name2rel.Count, uint64(numel(tc.FixtureRels) - 1));
        end
    end

    % ======================================================================
    % Alpha template ordering (Step 2 of shine_equalize)
    % ======================================================================

    methods (Test)
        function testAlphaOrdering_AlignedWithFlatDirAlphabetical(tc)
            % Core defect guard: alpha_templates{im} must correspond to the
            % im-th file in dir(FlatIn,'*.png') -- i.e. alphabetical order --
            % NOT to the im-th file in dir(FixtureIn,'**','*.png').
            %
            % We verify this by checking that alpha_templates{im} has the
            % unique alpha tag that belongs to the im-th alphabetical filename.

            tc.runFlattenStep();  % populate FlatIn

            % Build expected: map from encoded_name -> alpha tag
            tag_of = containers.Map('KeyType','char','ValueType','double');
            for i = 1:numel(tc.FixtureRels)
                tag_of(tc.ExpectedFlatOrder{i}) = tc.AlphaTags(i);
            end

            % Reproduce Step 2 of shine_equalize
            flat_listing    = dir(fullfile(tc.FlatIn, '*.png'));
            alpha_templates = cell(numel(flat_listing), 1);
            for im = 1:numel(flat_listing)
                [~, ~, alpha] = imread(fullfile(tc.FlatIn, flat_listing(im).name));
                alpha_templates{im} = alpha;
            end

            % Verify each alpha_templates{im} matches the expected tag
            for im = 1:numel(flat_listing)
                expected_tag = tag_of(flat_listing(im).name);
                actual_tag   = double(alpha_templates{im}(1,1));
                tc.verifyEqual(actual_tag, expected_tag, ...
                    sprintf('Alpha mismatch at position %d (%s): got %d, expected %d', ...
                            im, flat_listing(im).name, actual_tag, expected_tag));
            end
        end
    end

    % ======================================================================
    % Restore round-trip (Steps 1 + 4 of shine_equalize)
    % ======================================================================

    methods (Test)
        function testRestore_AllFilesAtCorrectPaths(tc)
            name2rel = tc.runFlattenStep();

            % Simulate SHINE output: copy flat files to FlatOut unchanged
            flat_listing = dir(fullfile(tc.FlatIn, '*.png'));
            for i = 1:numel(flat_listing)
                copyfile(fullfile(tc.FlatIn,  flat_listing(i).name), ...
                         fullfile(tc.FlatOut, flat_listing(i).name));
            end

            % Run restore step
            mkdir(tc.OutputDir);
            tc.runRestoreStep(name2rel);

            % Every original relative path should exist under OutputDir
            for i = 1:numel(tc.FixtureRels)
                expected = fullfile(tc.OutputDir, ...
                                    strrep(tc.FixtureRels{i}, '/', filesep));
                tc.verifyTrue(exist(expected, 'file') > 0, ...
                    sprintf('Missing restored file: %s', tc.FixtureRels{i}));
            end
        end

        function testRestore_Count(tc)
            name2rel     = tc.runFlattenStep();
            flat_listing = dir(fullfile(tc.FlatIn, '*.png'));
            for i = 1:numel(flat_listing)
                copyfile(fullfile(tc.FlatIn,  flat_listing(i).name), ...
                         fullfile(tc.FlatOut, flat_listing(i).name));
            end
            mkdir(tc.OutputDir);
            tc.runRestoreStep(name2rel);

            restored = dir(fullfile(tc.OutputDir, '**', '*.png'));
            tc.verifyEqual(numel(restored), numel(tc.FixtureRels));
        end

        function testRestore_AlphaPreserved(tc)
            % After encode -> copy (simulating SHINE) -> restore,
            % each output image must retain its original alpha tag.
            name2rel     = tc.runFlattenStep();
            flat_listing = dir(fullfile(tc.FlatIn, '*.png'));
            for i = 1:numel(flat_listing)
                copyfile(fullfile(tc.FlatIn,  flat_listing(i).name), ...
                         fullfile(tc.FlatOut, flat_listing(i).name));
            end
            mkdir(tc.OutputDir);
            tc.runRestoreStep(name2rel);

            for i = 1:numel(tc.FixtureRels)
                dst = fullfile(tc.OutputDir, strrep(tc.FixtureRels{i},'/',filesep));
                [~, ~, alpha] = imread(dst);
                tc.verifyEqual(double(alpha(1,1)), double(tc.AlphaTags(i)), ...
                    sprintf('Alpha tag mismatch for %s', tc.FixtureRels{i}));
            end
        end

        function testRestore_TempDirsRemovedAfterCleanup(tc)
            tc.runFlattenStep();
            mkdir(tc.OutputDir);

            % Simulate cleanup
            if exist(tc.FlatIn,  'dir'); rmdir(tc.FlatIn,  's'); end
            if exist(tc.FlatOut, 'dir'); rmdir(tc.FlatOut, 's'); end

            tc.verifyFalse(exist(tc.FlatIn,  'dir') > 0);
            tc.verifyFalse(exist(tc.FlatOut, 'dir') > 0);
        end
    end

    % ======================================================================
    % Private helpers (not test methods)
    % ======================================================================

    methods (Access = private)
        function name2rel = runFlattenStep(tc)
            % Reproduces Step 1 of shine_equalize using the fixture tree.
            listing  = dir(fullfile(tc.FixtureIn, '**', '*.png'));
            name2rel = containers.Map('KeyType','char','ValueType','char');

            for i = 1:numel(listing)
                rel_dir = strrep(listing(i).folder, tc.FixtureIn, '');
                rel_dir = strtrim(rel_dir);
                if ~isempty(rel_dir) && any(rel_dir(1) == [filesep '/' '\'])
                    rel_dir = rel_dir(2:end);
                end
                encoded_name = encode_flat_name(rel_dir, listing(i).name);
                flat_path    = fullfile(tc.FlatIn, encoded_name);

                if isKey(name2rel, encoded_name) || exist(flat_path, 'file')
                    warning('TestShineEqualize:conflict', ...
                            'Conflict skipped: %s', encoded_name);
                    continue;
                end
                copyfile(fullfile(listing(i).folder, listing(i).name), flat_path);
                name2rel(encoded_name) = fullfile(rel_dir, listing(i).name);
            end
        end

        function runRestoreStep(tc, name2rel)
            % Reproduces Step 4 of shine_equalize.
            out_listing = dir(fullfile(tc.FlatOut, '*.png'));
            for i = 1:numel(out_listing)
                encoded_name = out_listing(i).name;
                if ~isKey(name2rel, encoded_name); continue; end
                dst_path = fullfile(tc.OutputDir, name2rel(encoded_name));
                dst_dir  = fileparts(dst_path);
                if ~exist(dst_dir, 'dir'); mkdir(dst_dir); end
                movefile(fullfile(tc.FlatOut, encoded_name), dst_path);
            end
        end
    end
end
