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

// ── canSatisfyTrialRepeatSeparation ────────────────────────────────────────────
describe('canSatisfyTrialRepeatSeparation', () => {
    it('is always satisfiable with 0 repeats', () => {
        assert.equal(canSatisfyTrialRepeatSeparation(10, 0, 1000), true);
    });
    it('is satisfiable for a small separation', () => {
        assert.equal(canSatisfyTrialRepeatSeparation(10, 2, 2), true);
    });
    it('is unsatisfiable once separation exceeds t', () => {
        assert.equal(canSatisfyTrialRepeatSeparation(10, 2, 10), false);
    });
    it('matches the known-infeasible case from insertTrialRepeats tests (t=4, numRepeats=2, minSep=3)', () => {
        assert.equal(canSatisfyTrialRepeatSeparation(4, 2, 3), false);
    });
});

// ── verifyConfig ──────────────────────────────────────────────────────────────
describe('verifyConfig', () => {
    // Minimal valid config that passes all checks.
    const validConfig = () => ({
        stimuli_paths: {
            main_root: 'images',
            practice:  'practice',
            catch:     'catch',
        },
        design: {
            trials_per_subject:          10,
            images_per_trial:            20,
            frac_trials_repeated:        0,
            min_trial_repeat_separation: 2,
            min_trial_duration_ms:       5000,
            min_pairwise_distance_sd:    0.04,
            min_move_item_ratio:         0.75,
        },
        catch_trials: {
            num_trials:         2,
            images_per_trial:   10,
            location_tolerance: 0.20,
        },
        display: {
            sort_area_width:      900,
            sort_area_height:     700,
            sort_area_min_width:  900,
            sort_area_min_height: 700,
            sort_area_shape:      'rect',
            stim_starts_inside:   true,
            column_spread_factor: 0.3,
            image_size_fraction:  0.11,
            background_color:     '#808080',
            text_color:           '#111111',
            font_family:          'sans-serif',
            font_size:            '16px',
            line_height:          1.6,
        },
        deployment: {
            mode:                    'debug',
            debug_shine_variant:     'pre',
            prolific_completion_url: 'https://example.com',
            prolific_no_consent_url: 'https://example.com/no-consent',
            version:                 '2.2',
        },
        consent: {
            researcher_name:        'Test Researcher',
            researcher_email:       'test@example.com',
            pi_name:                'Test PI',
            pi_email:               'pi@example.com',
            lab_name:               'Test Lab',
            lab_phone:              '+1-555-0000',
            institution:            'Test University',
            study_duration_minutes: 45,
        },
    });

    it('passes a valid config without throwing', () => {
        assert.doesNotThrow(() => verifyConfig(validConfig()));
    });

    it('throws on missing key', () => {
        const cfg = validConfig();
        delete cfg.design.images_per_trial;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "design\.images_per_trial"/ },
        );
    });

    it('throws on wrong type', () => {
        const cfg = validConfig();
        cfg.design.trials_per_subject = '10';  // string instead of number
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.trials_per_subject" must be a number/ },
        );
    });

    it('throws on out-of-range value (image_size_fraction >= 1)', () => {
        const cfg = validConfig();
        cfg.display.image_size_fraction = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"display\.image_size_fraction" must be in \(0, 1\)/ },
        );
    });

    it('throws when num_catch_trials >= trials_per_subject', () => {
        const cfg = validConfig();
        cfg.catch_trials.num_trials = 10; // equals trials_per_subject
        assert.throws(
            () => verifyConfig(cfg),
            { message: /catch_trials\.num_trials \(10\) must be < design\.trials_per_subject/ },
        );
    });

    it('throws when k images do not fit in a grid on the minimum canvas', () => {
        const cfg = validConfig();
        cfg.display.image_size_fraction = 0.5;  // stimSize = round(900*0.5)=450; 5 cols * 450 > 900
        assert.throws(
            () => verifyConfig(cfg),
            { message: /cannot fit in a .+ grid/ },
        );
    });

    it('throws on invalid sort_area_shape', () => {
        const cfg = validConfig();
        cfg.display.sort_area_shape = 'square'; // old value, now invalid
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"display\.sort_area_shape" must be "rect" or "ellipse"/ },
        );
    });

    it('throws on missing min_move_item_ratio', () => {
        const cfg = validConfig();
        delete cfg.design.min_move_item_ratio;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "design\.min_move_item_ratio"/ },
        );
    });

    it('throws when min_move_item_ratio is out of range (> 1)', () => {
        const cfg = validConfig();
        cfg.design.min_move_item_ratio = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    it('throws when min_move_item_ratio is 0', () => {
        const cfg = validConfig();
        cfg.design.min_move_item_ratio = 0;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    it('throws when frac_trials_repeated is out of range', () => {
        const cfg = validConfig();
        cfg.design.frac_trials_repeated = 1;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.frac_trials_repeated" must be in \[0, 1\)/ },
        );
    });

    it('warns (not throws) when frac_trials_repeated >= 0.4', () => {
        const cfg = validConfig();
        cfg.design.frac_trials_repeated = 0.4;
        const warnings = [];
        const origWarn = console.warn;
        console.warn = msg => warnings.push(msg);
        assert.doesNotThrow(() => verifyConfig(cfg));
        console.warn = origWarn;
        assert.ok(warnings.some(w => /frac_trials_repeated/.test(w) && /0\.4/.test(w)));
    });

    it('throws on missing min_trial_repeat_separation', () => {
        const cfg = validConfig();
        delete cfg.design.min_trial_repeat_separation;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "design\.min_trial_repeat_separation"/ },
        );
    });

    it('throws when min_trial_repeat_separation is not a positive integer', () => {
        const cfg = validConfig();
        cfg.design.min_trial_repeat_separation = 0;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.min_trial_repeat_separation" must be a positive integer/ },
        );
    });

    it('throws at init when min_trial_repeat_separation is too large to ever be satisfied', () => {
        const cfg = validConfig();
        // t=10, frac_trials_repeated=0.2 → numTrialRepeats=2; minSep=100 is
        // impossible regardless of which trials end up singles-only.
        cfg.design.frac_trials_repeated = 0.2;
        cfg.design.min_trial_repeat_separation = 100;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /design\.min_trial_repeat_separation \(100\) cannot be satisfied/ },
        );
    });

    it('does not check min_trial_repeat_separation feasibility when frac_trials_repeated is 0', () => {
        const cfg = validConfig();
        // No repeat slots are ever created, so an absurd separation is moot.
        cfg.design.frac_trials_repeated = 0;
        cfg.design.min_trial_repeat_separation = 1000;
        assert.doesNotThrow(() => verifyConfig(cfg));
    });

});

// ── computeMainQcFlag ─────────────────────────────────────────────────────────
describe('computeMainQcFlag', () => {
    const cfg = {
        design: { min_pairwise_distance_sd: 0.04, min_move_item_ratio: 0.75 },
    };

    it('passes when sd is high enough and moves are sufficient', () => {
        assert.equal(computeMainQcFlag(0.10, 15, 20, cfg), false);
    });

    it('flags when sd is below threshold (pile-up)', () => {
        assert.equal(computeMainQcFlag(0.02, 15, 20, cfg), true);
    });

    it('flags when moves are below ratio threshold', () => {
        // 0.75 * 20 = 15; 14 moves → flag
        assert.equal(computeMainQcFlag(0.10, 14, 20, cfg), true);
    });

    it('passes at exactly the move threshold', () => {
        // 0.75 * 20 = 15; exactly 15 moves → pass
        assert.equal(computeMainQcFlag(0.10, 15, 20, cfg), false);
    });

    it('flags when both sd and moves fail', () => {
        assert.equal(computeMainQcFlag(0.02, 5, 20, cfg), true);
    });

    it('catches threshold scales with numItems (catch trial size)', () => {
        // 0.75 * 10 = 7.5; 7 moves → flag, 8 moves → pass
        assert.equal(computeMainQcFlag(0.10, 7, 10, cfg), true);
        assert.equal(computeMainQcFlag(0.10, 8, 10, cfg), false);
    });
});

// ── computeLayout ─────────────────────────────────────────────────────────────
describe('computeLayout', () => {
    // Helper: build a minimal config with a display section.
    const cfg = (maxW, maxH, minW, minH, frac = 0.11, inside = true, spread = 1.0) => ({
        display: {
            sort_area_width:      maxW,
            sort_area_height:     maxH,
            sort_area_min_width:  minW,
            sort_area_min_height: minH,
            image_size_fraction:  frac,
            stim_starts_inside:   inside,
            column_spread_factor: spread,
        },
    });

    // ── stim_starts_inside: true (images start inside — full 85% fraction) ───
    it('inside=true, large screen — capped at configured max', () => {
        // floor(1920 * 0.85) = 1632 > max 900  →  capped at 900
        // floor(1080 * 0.70) = 756  > max 700  →  capped at 700
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(900, 700, 900, 700));
        assert.equal(sortW,    900);
        assert.equal(sortH,    700);
        assert.equal(stimSize,  99); // round(900 * 0.11)
    });

    it('inside=true, medium screen — viewport fraction wins between min and max', () => {
        // floor(1200 * 0.85) = 1020, clamp to [900, 1400] → 1020
        // floor(900  * 0.70) = 630,  clamp to [600, 900]  → 630
        const { sortW, sortH } = computeLayout(1200, 900, cfg(1400, 900, 900, 600));
        assert.equal(sortW, 1020);
        assert.equal(sortH,  630);
    });

    // ── stim_starts_inside: false (images start outside — viewport / (1+factor)) ──
    it('inside=false, large screen with factor=0.3 — capped at max', () => {
        // floor(1920 / 1.3) = 1476 > max 1400  →  capped at 1400
        const { sortW } = computeLayout(1920, 1080, cfg(1400, 900, 900, 500, 0.11, false, 0.3));
        assert.equal(sortW, 1400);
    });

    it('inside=false, medium screen with factor=1.0 — constrained by staging', () => {
        // floor(1456 / 2.0) = 728, clamp to [650, 1000] → 728
        const { sortW, stimSize } = computeLayout(1456, 816, cfg(1000, 750, 650, 500, 0.11, false, 1.0));
        assert.equal(sortW,    728);
        assert.equal(stimSize,  80); // round(728 * 0.11)
    });

    it('inside=false, small screen — floor wins even with staging constraint', () => {
        // floor(1000 / 2.0) = 500 < min 650  →  floored at 650
        const { sortW } = computeLayout(1000, 800, cfg(1000, 750, 650, 500, 0.11, false, 1.0));
        assert.equal(sortW, 650);
    });
});
