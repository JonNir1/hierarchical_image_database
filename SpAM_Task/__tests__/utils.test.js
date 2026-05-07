// Run: node --test SpAM_Task/__tests__/utils.test.js
'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const vm     = require('node:vm');
const fs     = require('node:fs');
const path   = require('node:path');

// Load utils.js as globals (mirrors <script> loading in the browser).
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '../js/utils.js'), 'utf8'));

// Simple deterministic LCG used as a stand-in for seedrandom.
const makeRng = (seed = 42) => {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
};

// ── hashString ────────────────────────────────────────────────────────────────
describe('hashString', () => {
    it('returns a non-negative integer', () => {
        const h = hashString('hello');
        assert.ok(Number.isInteger(h) && h >= 0);
    });
    it('is deterministic', () => {
        assert.equal(hashString('abc'), hashString('abc'));
    });
    it('empty string returns 5381 (djb2 initial value)', () => {
        assert.equal(hashString(''), 5381);
    });
});

// ── seededShuffle ─────────────────────────────────────────────────────────────
describe('seededShuffle', () => {
    it('does not mutate the input', () => {
        const arr = [1, 2, 3, 4, 5];
        seededShuffle(arr, makeRng());
        assert.deepEqual(arr, [1, 2, 3, 4, 5]);
    });
    it('result has the same elements as the input', () => {
        const arr    = [1, 2, 3, 4, 5];
        const result = seededShuffle(arr, makeRng());
        assert.deepEqual([...result].sort((a, b) => a - b), arr);
    });
    it('is deterministic with the same seed', () => {
        const arr = [1, 2, 3, 4, 5, 6, 7, 8];
        assert.deepEqual(seededShuffle(arr, makeRng(7)), seededShuffle(arr, makeRng(7)));
    });
});

// ── computePairwiseDistances ──────────────────────────────────────────────────
describe('computePairwiseDistances', () => {
    it('returns n*(n-1)/2 pairs', () => {
        const locs = [{src:'a',x:0,y:0},{src:'b',x:1,y:0},{src:'c',x:0,y:1},{src:'d',x:1,y:1}];
        assert.equal(computePairwiseDistances(locs, 900, 700).length, 6);
    });
    it('normalises by the diagonal (3-4-5 triangle)', () => {
        const locs = [{src:'a',x:0,y:0},{src:'b',x:300,y:400}];
        const [{distance}] = computePairwiseDistances(locs, 900, 700);
        assert.ok(Math.abs(distance - 500 / Math.sqrt(900**2 + 700**2)) < 1e-10);
    });
});

// ── computeSD ─────────────────────────────────────────────────────────────────
describe('computeSD', () => {
    it('returns 0 for fewer than 2 values', () => {
        assert.equal(computeSD([]), 0);
        assert.equal(computeSD([5]), 0);
    });
    it('uses n-1 denominator (sample SD)', () => {
        // [2,4,4,4,5,5,7,9]: mean=5, Σ(x-μ)²=32  →  SD = √(32/7)
        assert.ok(Math.abs(computeSD([2,4,4,4,5,5,7,9]) - Math.sqrt(32/7)) < 1e-10);
    });
});

// ── verifyConfig ──────────────────────────────────────────────────────────────
describe('verifyConfig', () => {
    // Minimal valid config that passes all checks.
    const validConfig = () => ({
        trials_per_subject:        10,
        images_per_trial:          20,
        unique_images_per_subject: 150,
        num_catch_trials:          2,
        catch_images_per_trial:    10,
        practice_images_per_trial: 8,
        sort_area_width:           900,
        sort_area_height:          700,
        sort_area_min_width:       900,
        sort_area_min_height:      700,
        image_size_fraction:       0.11,
        min_trial_rt_ms:           5000,
        min_pairwise_distance_sd:  0.04,
        catch_cluster_max_mean:    0.15,
        catch_cluster_max_sd:      0.10,
        catch_location_tolerance:  0.20,
        sort_area_shape:           'square',
        debug:                     true,   // suppress deployment warnings
        stimuli_path:              'stimuli',
        stimuli_practice_path:     'practice',
        stimuli_catch_path:        'catch',
        prolific_completion_url:   'https://example.com',
    });

    it('passes a valid config without throwing', () => {
        assert.doesNotThrow(() => verifyConfig(validConfig()));
    });

    it('throws on missing key', () => {
        const cfg = validConfig();
        delete cfg.images_per_trial;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "images_per_trial"/ },
        );
    });

    it('throws on wrong type', () => {
        const cfg = validConfig();
        cfg.trials_per_subject = '10';  // string instead of number
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"trials_per_subject" must be a number/ },
        );
    });

    it('throws on out-of-range value (image_size_fraction >= 1)', () => {
        const cfg = validConfig();
        cfg.image_size_fraction = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"image_size_fraction" must be in \(0, 1\)/ },
        );
    });

    it('throws when unique_images_per_subject > t * k', () => {
        const cfg = validConfig();
        cfg.unique_images_per_subject = 201; // > 10*20
        assert.throws(
            () => verifyConfig(cfg),
            { message: /unique_images_per_subject \(201\) exceeds/ },
        );
    });

    it('throws when num_catch_trials >= trials_per_subject', () => {
        const cfg = validConfig();
        cfg.num_catch_trials = 10; // equals trials_per_subject
        assert.throws(
            () => verifyConfig(cfg),
            { message: /num_catch_trials \(10\) must be < trials_per_subject/ },
        );
    });

    it('throws when k images do not fit in a grid on the minimum canvas', () => {
        const cfg = validConfig();
        cfg.image_size_fraction = 0.5;  // stimSize = round(900*0.5)=450; 5 cols * 450 > 900
        assert.throws(
            () => verifyConfig(cfg),
            { message: /cannot fit in a .+ grid/ },
        );
    });

    it('warns (not throws) when practice_images_per_trial > images_per_trial', () => {
        const cfg = validConfig();
        cfg.practice_images_per_trial = 25; // > images_per_trial=20
        // Should not throw
        assert.doesNotThrow(() => verifyConfig(cfg));
    });
});

// ── computeLayout ─────────────────────────────────────────────────────────────
describe('computeLayout', () => {
    // Helper: build a minimal config with explicit min/max and fraction.
    const cfg = (maxW, maxH, minW, minH, frac = 0.11) => ({
        sort_area_width:    maxW,
        sort_area_height:   maxH,
        sort_area_min_width:  minW,
        sort_area_min_height: minH,
        image_size_fraction:  frac,
    });

    it('large screen — capped at configured max (900×700)', () => {
        // floor(1920 * 0.85) = 1632 > max 900  →  capped at 900
        // floor(1080 * 0.78) = 842  > max 700  →  capped at 700
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(900, 700, 900, 700));
        assert.equal(sortW,    900);
        assert.equal(sortH,    700);
        assert.equal(stimSize,  99); // round(900 * 0.11)
    });

    it('just-fits width — viewport slightly above min still yields min', () => {
        // floor(1060 * 0.85) = 901 > max 900  →  capped at 900
        // floor(900  * 0.78) = 702 > max 700  →  capped at 700
        const { sortW, sortH } = computeLayout(1060, 900, cfg(900, 700, 900, 700));
        assert.equal(sortW, 900);
        assert.equal(sortH, 700);
    });

    it('small screen with default mins — floor holds at 900×700', () => {
        // 800×600: floor(800*0.85)=680 < min 900; floor(600*0.78)=468 < min 700 → floor wins
        const { sortW, sortH, stimSize } = computeLayout(800, 600, cfg(900, 700, 900, 700));
        assert.equal(sortW,    900);
        assert.equal(sortH,    700);
        assert.equal(stimSize,  99);
    });

    it('small screen with low mins — shrinks to viewport-derived size', () => {
        // floor(700 * 0.85) = 595;  floor(500 * 0.78) = 390
        // both above min (400×350) and below max (900×700)  →  viewport wins
        const { sortW, sortH, stimSize } = computeLayout(700, 500, cfg(900, 700, 400, 350));
        assert.equal(sortW,    595);
        assert.equal(sortH,    390);
        assert.equal(stimSize,  65); // round(595 * 0.11)
    });

    it('large screen with high max — caps at the higher configured max', () => {
        // floor(1920 * 0.85) = 1632 > max 1100  →  capped at 1100
        // floor(1080 * 0.78) = 842  > max 800   →  capped at 800
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(1100, 800, 600, 500));
        assert.equal(sortW,    1100);
        assert.equal(sortH,    800);
        assert.equal(stimSize, 121); // round(1100 * 0.11)
    });
});
