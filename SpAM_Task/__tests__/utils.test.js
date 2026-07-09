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
        design: {
            num_blocks:                   4,
            trials_per_block:             5,
            repeats_per_block:            1,
            images_per_trial:             20,
            min_trial_repeat_separation:  2,
            min_trial_duration_ms:        5000,
            stimuli_path:                 'images',
        },
        catch_trials: {
            catch_per_block:     2,
            images_per_trial:    10,
            location_tolerance:  0.20,
            stimuli_path:        'catch',
        },
        screening: {
            trial_min_move_item_ratio:      0.75,
            trial_min_pairwise_distance_sd: 0.04,
            max_move_ratio_fail_frac:       0.30,
            max_distance_sd_fail_frac:      0.30,
            min_median_reliability:         0.30,
        },
        display: {
            sort_area_min_width:  900,
            sort_area_min_height: 700,
            min_header_height_px: 100,
            min_footer_height_px: 80,
            image_size_fraction:  0.11,
            background_color:     '#808080',
            text_color:           '#111111',
            font_family:          'sans-serif',
            font_size:            '16px',
            line_height:          1.6,
        },
        deployment: {
            mode:                'debug',
            debug_shine_variant: 'pre',
            version:             '4.0',
        },
        prolific: {
            completion_url:  'https://example.com',
            no_consent_url:  'https://example.com/no-consent',
            partial_completion_urls: ['https://example.com/p1', 'https://example.com/p2', 'https://example.com/p3'],
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
        cfg.design.num_blocks = '4';  // string instead of number
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.num_blocks" must be a number/ },
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

    it('throws when k images do not fit in a grid on the minimum canvas', () => {
        const cfg = validConfig();
        cfg.display.image_size_fraction = 0.5;  // stimSize = round(900*0.5)=450; 5 cols * 450 > 900
        assert.throws(
            () => verifyConfig(cfg),
            { message: /cannot fit in a .+ grid/ },
        );
    });

    it('throws on missing screening.trial_min_move_item_ratio', () => {
        const cfg = validConfig();
        delete cfg.screening.trial_min_move_item_ratio;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "screening\.trial_min_move_item_ratio"/ },
        );
    });

    it('throws when screening.trial_min_move_item_ratio is out of range (> 1)', () => {
        const cfg = validConfig();
        cfg.screening.trial_min_move_item_ratio = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening\.trial_min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    it('throws when screening.trial_min_move_item_ratio is 0', () => {
        const cfg = validConfig();
        cfg.screening.trial_min_move_item_ratio = 0;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening\.trial_min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    it('throws when repeats_per_block is out of range (negative)', () => {
        const cfg = validConfig();
        cfg.design.repeats_per_block = -1;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.repeats_per_block" must be an integer in \[0, design\.trials_per_block\]/ },
        );
    });

    it('throws when repeats_per_block exceeds trials_per_block', () => {
        const cfg = validConfig();
        cfg.design.repeats_per_block = cfg.design.trials_per_block + 1;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"design\.repeats_per_block" must be an integer in \[0, design\.trials_per_block\]/ },
        );
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

    it('throws at init when min_trial_repeat_separation cannot be satisfied within a single block', () => {
        const cfg = validConfig();
        // trials_per_block=5, repeats_per_block=1 -> tBlock=6; minSep=100 is
        // impossible regardless of which trial ends up single.
        cfg.design.min_trial_repeat_separation = 100;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /design\.min_trial_repeat_separation \(100\) cannot be satisfied within a single block/ },
        );
    });

    it('does not check min_trial_repeat_separation feasibility when repeats_per_block is 0', () => {
        const cfg = validConfig();
        // No repeat slots are ever created, so an absurd separation is moot.
        cfg.design.repeats_per_block = 0;
        cfg.design.min_trial_repeat_separation = 1000;
        assert.doesNotThrow(() => verifyConfig(cfg));
    });

    it('throws when partial_completion_urls length does not equal num_blocks - 1', () => {
        const cfg = validConfig();
        cfg.prolific.partial_completion_urls = ['only-one'];
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"prolific\.partial_completion_urls" must have length design\.num_blocks - 1 \(3\), got 1/ },
        );
    });

    it('throws when a partial_completion_urls element is not a string', () => {
        const cfg = validConfig();
        cfg.prolific.partial_completion_urls = ['a', 42, 'c'];
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"prolific\.partial_completion_urls\[1\]" must be a string/ },
        );
    });

    it('accepts num_blocks === 1 with an empty partial_completion_urls array', () => {
        const cfg = validConfig();
        cfg.design.num_blocks = 1;
        cfg.prolific.partial_completion_urls = [];
        assert.doesNotThrow(() => verifyConfig(cfg));
    });

});

// ── computeMainQcFlag ─────────────────────────────────────────────────────────
describe('computeMainQcFlag', () => {
    const cfg = {
        screening: { trial_min_pairwise_distance_sd: 0.04, trial_min_move_item_ratio: 0.75 },
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
    // Helper: build a minimal config. Sort area fills WIDTH_FILL_FRACTION (0.97) of
    // viewport width and all remaining viewport height after reserving
    // min_header_height_px/min_footer_height_px, each floor-protected by
    // sort_area_min_width/height.
    const cfg = (minW, minH, headerPx, footerPx, frac = 0.11) => ({
        display: {
            sort_area_min_width:  minW,
            sort_area_min_height: minH,
            min_header_height_px: headerPx,
            min_footer_height_px: footerPx,
            image_size_fraction:  frac,
        },
    });

    it('large screen — fill formula wins over the floor on both axes', () => {
        // floor(1920 * 0.97) = 1862 > min 900 → 1862
        // 1080 - 120 - 100 = 860 > min 550 → 860
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(900, 550, 120, 100));
        assert.equal(sortW,    1862);
        assert.equal(sortH,     860);
        assert.equal(stimSize,  205); // round(1862 * 0.11)
    });

    it('small screen — floor wins over the fill formula on both axes', () => {
        // floor(900 * 0.97) = 873 < min 900 → 900
        // 700 - 120 - 100 = 480 < min 550 → 550
        const { sortW, sortH } = computeLayout(900, 700, cfg(900, 550, 120, 100));
        assert.equal(sortW, 900);
        assert.equal(sortH, 550);
    });

    it('header/footer budgets are subtracted from available height', () => {
        // 1000 - 150 - 130 = 720, well above the 200px floor
        const { sortH } = computeLayout(1200, 1000, cfg(200, 200, 150, 130));
        assert.equal(sortH, 720);
    });
});

// ── computeSpearmanCorrelation ───────────────────────────────────────────────
describe('computeSpearmanCorrelation', () => {
    const pairs = (specs) => specs.map(([a, b, d]) => ({ src1: a, src2: b, distance: d }));

    it('returns 1 for a perfectly monotonic (identical) relationship, order-independent', () => {
        const a = pairs([['a', 'b', 1], ['a', 'c', 2], ['b', 'c', 3]]);
        // Same logical pairs, different array order and src1/src2 swapped on one entry.
        const b = pairs([['c', 'b', 3], ['a', 'c', 2], ['a', 'b', 1]]);
        assert.equal(computeSpearmanCorrelation(a, b), 1);
    });

    it('returns -1 for a perfectly reversed relationship', () => {
        const a = pairs([['a', 'b', 1], ['a', 'c', 2], ['b', 'c', 3]]);
        const b = pairs([['a', 'b', 3], ['a', 'c', 2], ['b', 'c', 1]]);
        assert.equal(computeSpearmanCorrelation(a, b), -1);
    });

    it('handles tied ranks via average-rank Spearman (known-answer)', () => {
        // Classic textbook vector: values [1,2,2,3] vs [4,3,2,1] (paired by index/key).
        const a = pairs([['w','x',1], ['w','y',2], ['w','z',2], ['x','y',3]]);
        const b = pairs([['w','x',4], ['w','y',3], ['w','z',2], ['x','y',1]]);
        // Ranks of a: [1, 2.5, 2.5, 4]; ranks of b: [4, 3, 2, 1].
        // Pearson correlation of those two rank vectors:
        const rankA = [1, 2.5, 2.5, 4], rankB = [4, 3, 2, 1];
        const meanA = rankA.reduce((s,v)=>s+v,0)/4, meanB = rankB.reduce((s,v)=>s+v,0)/4;
        let num=0, denA=0, denB=0;
        for (let i=0;i<4;i++){ const da=rankA[i]-meanA, db=rankB[i]-meanB; num+=da*db; denA+=da*da; denB+=db*db; }
        const expected = num / Math.sqrt(denA*denB);
        assert.ok(Math.abs(computeSpearmanCorrelation(a, b) - expected) < 1e-10);
    });

    it('throws when the two pair sets cover different image sets', () => {
        const a = pairs([['a', 'b', 1], ['a', 'c', 2]]);
        const b = pairs([['a', 'b', 1], ['a', 'd', 2]]);
        assert.throws(() => computeSpearmanCorrelation(a, b), { message: /present in pairsA but not pairsB|image sets differ/ });
    });

    it('returns NaN for zero-variance input', () => {
        const a = pairs([['a', 'b', 5], ['a', 'c', 5], ['b', 'c', 5]]);
        const b = pairs([['a', 'b', 1], ['a', 'c', 2], ['b', 'c', 3]]);
        assert.ok(Number.isNaN(computeSpearmanCorrelation(a, b)));
    });
});

// ── evaluateScreening ─────────────────────────────────────────────────────────
describe('evaluateScreening', () => {
    const cfg = {
        screening: {
            trial_min_move_item_ratio:      0.75,
            trial_min_pairwise_distance_sd: 0.04,
            max_move_ratio_fail_frac:       0.30,
            max_distance_sd_fail_frac:      0.30,
            min_median_reliability:         0.30,
        },
    };

    it('skips (does not fail) the reliability criterion when zero repeats completed so far', () => {
        const result = evaluateScreening({ mainTrials: [{ numMoves: 20, numItems: 20, sd: 0.1 }], reliabilities: [] }, cfg);
        assert.equal(result.stats.medianReliability, null);
        assert.ok(!result.reasons.some(r => /reliability/.test(r)));
        assert.equal(result.pass, true);
    });

    it('passes when the fail-fraction is exactly at the threshold (strict inequality)', () => {
        // 3/10 = 0.30 exactly equals max_move_ratio_fail_frac -> should pass
        const mainTrials = [
            ...Array.from({ length: 3 }, () => ({ numMoves: 1, numItems: 20, sd: 0.1 })), // fail move ratio
            ...Array.from({ length: 7 }, () => ({ numMoves: 20, numItems: 20, sd: 0.1 })),
        ];
        const result = evaluateScreening({ mainTrials, reliabilities: [] }, cfg);
        assert.equal(result.stats.moveRatioFailFrac, 0.3);
        assert.equal(result.pass, true);
    });

    it('fails when exactly one criterion is violated', () => {
        const mainTrials = Array.from({ length: 10 }, () => ({ numMoves: 1, numItems: 20, sd: 0.1 })); // all fail move ratio
        const result = evaluateScreening({ mainTrials, reliabilities: [] }, cfg);
        assert.equal(result.pass, false);
        assert.equal(result.reasons.length, 1);
        assert.ok(/move-ratio/.test(result.reasons[0]));
    });

    it('fails with multiple reasons when multiple criteria are violated', () => {
        const mainTrials = Array.from({ length: 10 }, () => ({ numMoves: 1, numItems: 20, sd: 0.01 })); // fail both
        const result = evaluateScreening({ mainTrials, reliabilities: [0.1, 0.05] }, cfg); // also fail reliability
        assert.equal(result.pass, false);
        assert.equal(result.reasons.length, 3);
    });

    it('does not divide by zero with an empty mainTrials array', () => {
        const result = evaluateScreening({ mainTrials: [], reliabilities: [] }, cfg);
        assert.equal(result.stats.moveRatioFailFrac, 0);
        assert.equal(result.stats.distanceSdFailFrac, 0);
        assert.equal(result.pass, true);
    });
});
