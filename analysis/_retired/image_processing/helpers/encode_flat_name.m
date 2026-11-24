function encoded = encode_flat_name(rel_dir, filename)
% encode_flat_name  Encode a relative subdirectory path into a flat filename.
%
%   encoded = encode_flat_name(rel_dir, filename)
%
%   Replaces all path separators in REL_DIR with '__' and prepends the result
%   to FILENAME.  If REL_DIR is empty, FILENAME is returned unchanged.
%
%   This is the inverse of the decode step in shine_equalize's restore loop:
%   given the Map entry  encoded_name -> rel_subpath, the rel_subpath is
%   simply  fullfile(rel_dir, filename).
%
%   Examples
%     encode_flat_name('animate\animal\body\bird', 'sparrow.png')
%       -> 'animate__animal__body__bird__sparrow.png'
%     encode_flat_name('', 'top_level.png')
%       -> 'top_level.png'

if isempty(rel_dir)
    encoded = filename;
else
    enc = strrep(rel_dir, filesep, '__');
    enc = strrep(enc, '/',  '__');
    enc = strrep(enc, '\',  '__');
    encoded = [enc '__' filename];
end
end
