// Run: node --test SpAM_Task/__tests__/trial_generator.test.js
'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const vm     = require('node:vm');
const fs     = require('node:fs');
const path   = require('node:path');

// Load globals in dependency order (mirrors <script> tags).
const load = f => vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, f), 'utf8'));
load('../js/utils.js');
load('../js/trial_generator.js');

const makeRng = (seed = 42) => {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
};

const ALL_IMAGES = Array.from({ length: 754 }, (_, i) => `img_${i}.png`);
// r=1/3 → N = round(200 / 1.333) = 150; n_double = 200 - 150 = 50
const CONFIG     = { design: { trials_per_subject: 10, images_per_trial: 20, frac_images_repeated: 1/3 } };

// ── buildTrialLists ───────────────────────────────────────────────────────────
describe('buildTrialLists', () => {
    it('returns t trials each with exactly k images', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        assert.equal(trials.length, 10);
        trials.forEach(t => assert.equal(t.length, 20));
    });
    it('no duplicates within a single trial', () => {
        buildTrialLists(ALL_IMAGES, CONFIG, makeRng())
            .forEach(t => assert.equal(new Set(t).size, t.length));
    });
    it('exactly n_double images appear in 2 trials; the rest appear in 1', () => {
        const counts = {};
        buildTrialLists(ALL_IMAGES, CONFIG, makeRng())
            .flat().forEach(img => { counts[img] = (counts[img] || 0) + 1; });
        const freq = Object.values(counts);
        assert.equal(freq.filter(c => c === 2).length, 50);
        assert.ok(freq.every(c => c === 1 || c === 2));
    });
    it('is deterministic with the same seed', () => {
        assert.deepEqual(
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7)),
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7))
        );
    });
    it('throws when the image pool cannot fill all trials', () => {
        // 2 images, 3 trials × 2 slots — impossible to fill every trial
        // r=0 → N = round(6/1) = 6, but pool only has 2 images
        assert.throws(() =>
            buildTrialLists(['x.png','y.png'],
                { design: { trials_per_subject:3, images_per_trial:2, frac_images_repeated:0 } },
                makeRng())
        );
    });
});

