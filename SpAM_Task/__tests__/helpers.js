// Shared fixtures for the SpAM_Task node:test files (not itself a test file).
'use strict';

const vm   = require('node:vm');
const fs   = require('node:fs');
const path = require('node:path');

// Load a script as globals, relative to this directory (mirrors <script> loading in the browser).
const load = f => vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, f), 'utf8'));

// Simple deterministic LCG used as a stand-in for seedrandom.
const makeRng = (seed = 42) => {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
};

const ALL_IMAGES = Array.from({ length: 754 }, (_, i) => `img_${i}.png`);

const CONFIG = {
    experimental_trials: {
        images_per_trial: 20,
    },
    screening_block: {
        enabled: true,
        num_experimental_trials: 5,
        num_repeat_trials:       0,
        min_repeat_separation:   2,
        num_catch_trials:        1,
    },
    experimental_block: {
        num_experimental_trials: 5,
        num_repeat_trials:       0,
        min_repeat_separation:   2,
        num_catch_trials:        1,
    },
    catch_trials: {
        images_per_trial: 10,
    },
};

module.exports = { load, makeRng, ALL_IMAGES, CONFIG };
