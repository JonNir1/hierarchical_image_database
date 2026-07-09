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
        experimental_trials: {
            stimuli_path:              'images',
            images_per_trial:          20,
            min_trial_duration_ms:     5000,
            min_move_item_ratio:       0.75,
            min_pairwise_distance_sd:  0.04,
        },
        catch_trials: {
            stimuli_path:        'catch',
            images_per_trial:    10,
            location_tolerance:  0.20,
        },
        screening_block: {
            enabled:                  true,
            prolific_code:            'SCREENCODE1',
            num_catch_trials:         1,
            num_experimental_trials:  5,
            num_repeat_trials:        1,
            min_repeat_separation:    2,
            thresholds: {
                min_reliability:           0.30,
                median_reliability:        null,
                move_ratio_max_fail_rate:  0.30,
                distance_sd_max_fail_rate: 0.30,
            },
        },
        experimental_block: {
            prolific_code:            'EXPCODE1',
            num_catch_trials:         2,
            num_experimental_trials:  15,
            num_repeat_trials:        0,
            min_repeat_separation:    3,
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
        consent: {
            researcher_name:        'Test Researcher',
            researcher_email:       'test@example.com',
            pi_name:                'Test PI',
            pi_email:               'pi@example.com',
            lab_name:               'Test Lab',
            lab_phone:              '+1-555-0000',
            institution:            'Test University',
            study_duration_minutes: 45,
            no_consent_code:        'NOCONSENT1',
        },
    });

    it('passes a valid config without throwing', () => {
        assert.doesNotThrow(() => verifyConfig(validConfig()));
    });

    it('throws on missing key', () => {
        const cfg = validConfig();
        delete cfg.experimental_trials.images_per_trial;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "experimental_trials\.images_per_trial"/ },
        );
    });

    it('throws on wrong type', () => {
        const cfg = validConfig();
        cfg.screening_block.num_experimental_trials = '5';  // string instead of number
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening_block\.num_experimental_trials" must be a number/ },
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

    it('throws on missing experimental_trials.min_move_item_ratio', () => {
        const cfg = validConfig();
        delete cfg.experimental_trials.min_move_item_ratio;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "experimental_trials\.min_move_item_ratio"/ },
        );
    });

    it('throws when experimental_trials.min_move_item_ratio is out of range (> 1)', () => {
        const cfg = validConfig();
        cfg.experimental_trials.min_move_item_ratio = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"experimental_trials\.min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    it('throws when experimental_trials.min_move_item_ratio is 0', () => {
        const cfg = validConfig();
        cfg.experimental_trials.min_move_item_ratio = 0;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"experimental_trials\.min_move_item_ratio" must be in \(0, 1\]/ },
        );
    });

    for (const stageName of ['screening_block', 'experimental_block']) {
        it(`throws when ${stageName}.num_repeat_trials is out of range (negative)`, () => {
            const cfg = validConfig();
            cfg[stageName].num_repeat_trials = -1;
            assert.throws(
                () => verifyConfig(cfg),
                { message: new RegExp(`"${stageName}\\.num_repeat_trials" must be an integer in \\[0, ${stageName}\\.num_experimental_trials\\]`) },
            );
        });

        it(`throws when ${stageName}.num_repeat_trials exceeds num_experimental_trials`, () => {
            const cfg = validConfig();
            cfg[stageName].num_repeat_trials = cfg[stageName].num_experimental_trials + 1;
            assert.throws(
                () => verifyConfig(cfg),
                { message: new RegExp(`"${stageName}\\.num_repeat_trials" must be an integer in \\[0, ${stageName}\\.num_experimental_trials\\]`) },
            );
        });

        it(`throws on missing ${stageName}.min_repeat_separation`, () => {
            const cfg = validConfig();
            delete cfg[stageName].min_repeat_separation;
            assert.throws(
                () => verifyConfig(cfg),
                { message: new RegExp(`missing required key "${stageName}\\.min_repeat_separation"`) },
            );
        });

        it(`allows ${stageName}.min_repeat_separation of 0 (loosened floor)`, () => {
            const cfg = validConfig();
            cfg[stageName].min_repeat_separation = 0;
            assert.doesNotThrow(() => verifyConfig(cfg));
        });

        it(`throws when ${stageName}.min_repeat_separation is negative`, () => {
            const cfg = validConfig();
            cfg[stageName].min_repeat_separation = -1;
            assert.throws(
                () => verifyConfig(cfg),
                { message: new RegExp(`"${stageName}\\.min_repeat_separation" must be a non-negative integer`) },
            );
        });

        it(`does not check ${stageName} min_repeat_separation feasibility when num_repeat_trials is 0`, () => {
            const cfg = validConfig();
            cfg[stageName].num_repeat_trials = 0;
            cfg[stageName].min_repeat_separation = 1000;
            assert.doesNotThrow(() => verifyConfig(cfg));
        });
    }

    it('throws at init when screening_block.min_repeat_separation cannot be satisfied within screening_block', () => {
        const cfg = validConfig();
        // num_experimental_trials=5, num_repeat_trials=1 -> t=6; minSep=100 is impossible.
        cfg.screening_block.min_repeat_separation = 100;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /screening_block\.min_repeat_separation \(100\) cannot be satisfied within screening_block/ },
        );
    });

    it('throws at init when experimental_block.min_repeat_separation cannot be satisfied within experimental_block', () => {
        const cfg = validConfig();
        cfg.experimental_block.num_repeat_trials = 3;
        cfg.experimental_block.min_repeat_separation = 1000;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /experimental_block\.min_repeat_separation \(1000\) cannot be satisfied within experimental_block/ },
        );
    });

    it('skips screening_block feasibility check entirely when disabled, even if infeasible', () => {
        const cfg = validConfig();
        cfg.screening_block.enabled = false;
        cfg.screening_block.min_repeat_separation = 1000; // would be infeasible if enabled
        assert.doesNotThrow(() => verifyConfig(cfg));
    });

    it('still validates screening_block numeric fields even when disabled', () => {
        const cfg = validConfig();
        cfg.screening_block.enabled = false;
        cfg.screening_block.num_experimental_trials = -1;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening_block\.num_experimental_trials" must be a positive integer/ },
        );
    });

    it('throws on missing screening_block.thresholds', () => {
        const cfg = validConfig();
        delete cfg.screening_block.thresholds;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required section "screening_block\.thresholds"/ },
        );
    });

    for (const key of ['min_reliability', 'median_reliability', 'move_ratio_max_fail_rate', 'distance_sd_max_fail_rate']) {
        it(`accepts null for screening_block.thresholds.${key}`, () => {
            const cfg = validConfig();
            cfg.screening_block.thresholds[key] = null;
            assert.doesNotThrow(() => verifyConfig(cfg));
        });

        it(`throws when screening_block.thresholds.${key} is a string`, () => {
            const cfg = validConfig();
            cfg.screening_block.thresholds[key] = 'not-a-number';
            assert.throws(
                () => verifyConfig(cfg),
                { message: new RegExp(`"screening_block\\.thresholds\\.${key}" must be a number or null`) },
            );
        });
    }

    it('throws when screening_block.thresholds.min_reliability is out of [-1,1]', () => {
        const cfg = validConfig();
        cfg.screening_block.thresholds.min_reliability = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening_block\.thresholds\.min_reliability" must be in \[-1,1\] or null/ },
        );
    });

    it('throws when screening_block.thresholds.move_ratio_max_fail_rate is out of [0,1]', () => {
        const cfg = validConfig();
        cfg.screening_block.thresholds.move_ratio_max_fail_rate = 1.5;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /"screening_block\.thresholds\.move_ratio_max_fail_rate" must be in \[0,1\] or null/ },
        );
    });

    it('throws on missing consent.no_consent_code', () => {
        const cfg = validConfig();
        delete cfg.consent.no_consent_code;
        assert.throws(
            () => verifyConfig(cfg),
            { message: /missing required key "consent\.no_consent_code"/ },
        );
    });

});

// ── computeMainQcFlag ─────────────────────────────────────────────────────────
describe('computeMainQcFlag', () => {
    const cfg = {
        experimental_trials: { min_pairwise_distance_sd: 0.04, min_move_item_ratio: 0.75 },
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

    it('matches an externally-verified (scipy spearmanr) mock-distance value', () => {
        // Mock pairwise distances for 8 image-pairs with no ties, one shuffled
        // and relabeled (src1/src2 swapped, different array order) to also
        // exercise unordered-pair matching. Expected value cross-checked via
        // scipy.stats.spearmanr(a, b) == 0.9285714285714287 (not derived from
        // this codebase's own implementation).
        const keys = [['a','b'], ['a','c'], ['a','d'], ['a','e'], ['a','f'], ['a','g'], ['a','h'], ['b','c']];
        const distA = [0.12, 0.45, 0.33, 0.78, 0.21, 0.60, 0.05, 0.90];
        const distB = [0.20, 0.30, 0.50, 0.65, 0.40, 0.55, 0.10, 0.80];
        const a = keys.map(([s1, s2], i) => ({ src1: s1, src2: s2, distance: distA[i] }));
        // b: reversed order, src1/src2 swapped on every pair.
        const b = keys.map(([s1, s2], i) => ({ src1: s2, src2: s1, distance: distB[i] })).reverse();
        const rho = computeSpearmanCorrelation(a, b);
        assert.ok(Math.abs(rho - 0.9285714285714287) < 1e-10, `expected ~0.9285714285714287, got ${rho}`);
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
        experimental_trials: {
            min_move_item_ratio:      0.75,
            min_pairwise_distance_sd: 0.04,
        },
        screening_block: {
            thresholds: {
                min_reliability:           0.30,
                median_reliability:        0.30,
                move_ratio_max_fail_rate:  0.30,
                distance_sd_max_fail_rate: 0.30,
            },
        },
    };

    it('skips (does not fail) the reliability criteria when zero repeats completed so far', () => {
        const result = evaluateScreening({ mainTrials: [{ numMoves: 20, numItems: 20, sd: 0.1 }], reliabilities: [] }, cfg);
        assert.equal(result.stats.minReliability, null);
        assert.equal(result.stats.medianReliability, null);
        assert.ok(!result.reasons.some(r => /reliability/.test(r)));
        assert.equal(result.pass, true);
    });

    it('fails on the MINIMUM reliability, not an average/median — one bad repeat is enough', () => {
        // Median/mean of these would be well above threshold (0.3); the min (0.05) is not.
        const result = evaluateScreening({ mainTrials: [], reliabilities: [0.9, 0.95, 0.05] }, cfg);
        assert.equal(result.stats.minReliability, 0.05);
        assert.equal(result.pass, false);
        assert.ok(result.reasons.some(r => /minimum reliability/.test(r)));
    });

    it('median_reliability is independent of min_reliability — fails on median even when min passes threshold, and vice versa', () => {
        // min=0.05 fails min_reliability(0.30); median=0.9 passes median_reliability(0.30).
        const onlyMinFails = evaluateScreening({ mainTrials: [], reliabilities: [0.05, 0.9, 0.95] }, cfg);
        assert.ok(onlyMinFails.reasons.some(r => /minimum reliability/.test(r)));
        assert.ok(!onlyMinFails.reasons.some(r => /median reliability/.test(r)));

        // min=0.10 passes min_reliability(0.05 threshold below); median=0.15 fails median_reliability(0.30).
        const lenientCfg = { ...cfg, screening_block: { thresholds: { ...cfg.screening_block.thresholds, min_reliability: 0.05 } } };
        const onlyMedianFails = evaluateScreening({ mainTrials: [], reliabilities: [0.10, 0.15, 0.20] }, lenientCfg);
        assert.ok(!onlyMedianFails.reasons.some(r => /minimum reliability/.test(r)));
        assert.ok(onlyMedianFails.reasons.some(r => /median reliability/.test(r)));
    });

    it('null disables each threshold criterion individually, even when the underlying stat would otherwise fail it', () => {
        const allDisabled = {
            experimental_trials: cfg.experimental_trials,
            screening_block: { thresholds: { min_reliability: null, median_reliability: null, move_ratio_max_fail_rate: null, distance_sd_max_fail_rate: null } },
        };
        const mainTrials = Array.from({ length: 10 }, () => ({ numMoves: 1, numItems: 20, sd: 0.01 })); // would fail both rate criteria
        const result = evaluateScreening({ mainTrials, reliabilities: [0.01, 0.02] }, allDisabled); // would fail both reliability criteria
        assert.equal(result.pass, true);
        assert.equal(result.reasons.length, 0);
    });

    it('passes when every individual reliability is at or above the threshold', () => {
        const result = evaluateScreening({ mainTrials: [], reliabilities: [0.9, 0.5, 0.30] }, cfg);
        assert.equal(result.stats.minReliability, 0.30);
        assert.equal(result.pass, true); // exactly-at-threshold passes (strict inequality)
    });

    it('passes when the fail-rate is exactly at the threshold (strict inequality)', () => {
        // 3/10 = 0.30 exactly equals move_ratio_max_fail_rate -> should pass
        const mainTrials = [
            ...Array.from({ length: 3 }, () => ({ numMoves: 1, numItems: 20, sd: 0.1 })), // fail move ratio
            ...Array.from({ length: 7 }, () => ({ numMoves: 20, numItems: 20, sd: 0.1 })),
        ];
        const result = evaluateScreening({ mainTrials, reliabilities: [] }, cfg);
        assert.equal(result.stats.moveRatioFailRate, 0.3);
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
        const result = evaluateScreening({ mainTrials, reliabilities: [0.1, 0.05] }, cfg); // also fail both reliability criteria
        assert.equal(result.pass, false);
        assert.equal(result.reasons.length, 4);
    });

    it('does not divide by zero with an empty mainTrials array', () => {
        const result = evaluateScreening({ mainTrials: [], reliabilities: [] }, cfg);
        assert.equal(result.stats.moveRatioFailRate, 0);
        assert.equal(result.stats.distanceSdFailRate, 0);
        assert.equal(result.pass, true);
    });
});
