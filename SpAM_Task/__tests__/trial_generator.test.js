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

// ── buildTrialLists ───────────────────────────────────────────────────────────
describe('buildTrialLists', () => {
    it('returns the sum of both stages\' num_experimental_trials, each with exactly images_per_trial images', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        assert.equal(trials.length, 10); // 5 (screening) + 5 (experimental)
        trials.forEach(t => assert.equal(t.length, 20));
    });
    it('no duplicates within a single trial', () => {
        buildTrialLists(ALL_IMAGES, CONFIG, makeRng())
            .forEach(t => assert.equal(new Set(t).size, t.length));
    });
    it('every image appears in at most one distinct trial', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        const counts = {};
        trials.flat().forEach(img => { counts[img] = (counts[img] || 0) + 1; });
        assert.ok(Object.values(counts).every(c => c === 1));
    });
    it('is deterministic with the same seed', () => {
        assert.deepEqual(
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7)),
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7))
        );
    });
    it('throws when the image pool cannot fill all trials', () => {
        // 2 images, screening (disabled, 0) + experimental (3) * images_per_trial=2 = 6 needed — impossible
        const cfg = {
            experimental_trials: { images_per_trial: 2 },
            screening_block:     { enabled: false, num_experimental_trials: 1 },
            experimental_block:  { num_experimental_trials: 3 },
        };
        assert.throws(() =>
            buildTrialLists(['x.png', 'y.png'], cfg, makeRng()),
            { message: /image pool has 2 image\(s\), need 6/ }
        );
    });
    it('sums to 0 screening trials when screening_block.enabled is false', () => {
        const cfg = { ...CONFIG, screening_block: { ...CONFIG.screening_block, enabled: false } };
        const trials = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.equal(trials.length, 5); // experimental_block only
    });
    it('returns the same total regardless of num_repeat_trials (additive, not substitutive)', () => {
        const cfg = { ...CONFIG, screening_block: { ...CONFIG.screening_block, num_repeat_trials: 2 } };
        const trials = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.equal(trials.length, 10); // unchanged — repeats are added on top, not carved out
    });
});

// ── partitionIntoStages ───────────────────────────────────────────────────────
describe('partitionIntoStages', () => {
    it('splits into a screening slice and an experimental slice, order preserved, no overlap', () => {
        const cfg = {
            screening_block:    { enabled: true, num_experimental_trials: 3 },
            experimental_block: { num_experimental_trials: 4 },
        };
        const distinctTrials = Array.from({ length: 7 }, (_, i) => [`t${i}_a`, `t${i}_b`]);
        const { screening, experimental } = partitionIntoStages(distinctTrials, cfg);
        assert.deepEqual(screening, distinctTrials.slice(0, 3));
        assert.deepEqual(experimental, distinctTrials.slice(3, 7));
    });

    it('screening slice is empty when screening_block.enabled is false', () => {
        const cfg = {
            screening_block:    { enabled: false, num_experimental_trials: 3 },
            experimental_block: { num_experimental_trials: 4 },
        };
        const distinctTrials = Array.from({ length: 4 }, (_, i) => [`t${i}`]);
        const { screening, experimental } = partitionIntoStages(distinctTrials, cfg);
        assert.deepEqual(screening, []);
        assert.deepEqual(experimental, distinctTrials);
    });

    it('slice lengths are independently correct when unequal (not uniform groups)', () => {
        const cfg = {
            screening_block:    { enabled: true, num_experimental_trials: 3 },
            experimental_block: { num_experimental_trials: 15 },
        };
        const distinctTrials = Array.from({ length: 18 }, (_, i) => [`t${i}`]);
        const { screening, experimental } = partitionIntoStages(distinctTrials, cfg);
        assert.equal(screening.length, 3);
        assert.equal(experimental.length, 15);
    });
});

// ── insertTrialRepeats ────────────────────────────────────────────────────────
describe('insertTrialRepeats', () => {
    it('returns trials unchanged (wrapped) when numRepeats is 0', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng()).slice(0, 5);
        const result = insertTrialRepeats(trials, 0, 2, 'screening', makeRng());
        assert.equal(result.length, 5);
        result.forEach((r, i) => {
            assert.equal(r.isRepeat, false);
            assert.equal(r.repeatOfTrialId, null);
            assert.equal(r.trialId, 'screening_' + i);
            assert.deepEqual(r.images, trials[i]);
        });
    });

    it('produces exactly numRepeats repeat entries with matching image sets', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng()).slice(0, 5); // 5 distinct
        const result = insertTrialRepeats(trials, 3, 2, 'screening', makeRng()); // 5 distinct + 3 repeats

        assert.equal(result.length, 8);
        const repeats = result.filter(r => r.isRepeat);
        assert.equal(repeats.length, 3);
        repeats.forEach(rep => {
            const original = result.find(r => r.trialId === rep.repeatOfTrialId);
            assert.ok(original);
            assert.deepEqual([...rep.images].sort(), [...original.images].sort());
        });
    });

    it('any distinct trial may be repeated (no eligibility restriction)', () => {
        const trials = [['a','b','c'], ['d','e','f'], ['g','h','i'], ['j','k','l']];
        const result = insertTrialRepeats(trials, 1, 1, 'screening', makeRng());
        assert.equal(result.filter(r => r.isRepeat).length, 1);
    });

    it('respects minSep (stage-local)', () => {
        // 7 distinct + 3 repeats -> t=10, minSep=3: known-feasible layout.
        const trials = buildTrialLists(ALL_IMAGES, {
            experimental_trials: { images_per_trial: 20 },
            screening_block: { enabled: false, num_experimental_trials: 0 },
            experimental_block: { num_experimental_trials: 7 },
        }, makeRng());
        const result = insertTrialRepeats(trials, 3, 3, 'experimental', makeRng());
        const positionOf = id => result.findIndex(r => r.trialId === id);
        result.forEach((rep, pos) => {
            if (rep.isRepeat) {
                assert.ok(pos - positionOf(rep.repeatOfTrialId) >= 3);
            }
        });
    });

    it('is deterministic with the same seed', () => {
        const run = seed => {
            const rng = makeRng(seed);
            const trials = buildTrialLists(ALL_IMAGES, CONFIG, rng).slice(0, 5);
            return insertTrialRepeats(trials, 3, 2, 'screening', rng);
        };
        assert.deepEqual(run(7), run(7));
    });

    it('namespaces trialId by stageLabel, disjoint across stages', () => {
        const trials = [['a','b'], ['c','d']];
        const resultScreening   = insertTrialRepeats(trials, 0, 1, 'screening', makeRng(1));
        const resultExperimental = insertTrialRepeats(trials, 0, 1, 'experimental', makeRng(1));
        assert.deepEqual(resultScreening.map(r => r.trialId), ['screening_0', 'screening_1']);
        assert.deepEqual(resultExperimental.map(r => r.trialId), ['experimental_0', 'experimental_1']);
    });

    it('throws when minSep cannot be satisfied within a stage', () => {
        // 2 distinct + 2 repeats -> t=4; minSep=3 impossible to fit 2 repeats with gap 3 in 4 slots
        const trials = buildTrialLists(ALL_IMAGES, {
            experimental_trials: { images_per_trial: 20 },
            screening_block: { enabled: false, num_experimental_trials: 0 },
            experimental_block: { num_experimental_trials: 2 },
        }, makeRng());
        assert.throws(
            () => insertTrialRepeats(trials, 2, 3, 'experimental', makeRng()),
            { message: /min_repeat_separation/ }
        );
    });
});

// ── buildStage ────────────────────────────────────────────────────────────────
describe('buildStage', () => {
    it('returns num_experimental_trials + num_repeat_trials + num_catch_trials trials, all stamped with the stage label', () => {
        const stageConfig = { num_repeat_trials: 1, min_repeat_separation: 2, num_catch_trials: 1 };
        const catchTrialsConfig = { images_per_trial: 10 };
        const distinctTrials = buildTrialLists(ALL_IMAGES, {
            experimental_trials: { images_per_trial: 20 },
            screening_block: { enabled: false, num_experimental_trials: 0 },
            experimental_block: { num_experimental_trials: 5 },
        }, makeRng());
        const catchPool = Array.from({ length: 50 }, (_, i) => `catch_${i}.png`);
        const combined = buildStage(distinctTrials, catchPool, stageConfig, catchTrialsConfig, makeRng(), 'experimental');

        assert.equal(combined.length, 5 + 1 + 1); // num_experimental_trials + num_repeat_trials + num_catch_trials
        combined.forEach(t => assert.equal(t.block, 'experimental'));

        const repeat = combined.find(t => t.type === 'main' && t.isRepeat);
        const original = combined.find(t => t.type === 'main' && t.trialId === repeat.repeatOfTrialId);
        assert.deepEqual([...repeat.images].sort(), [...original.images].sort());
    });
});

// ── Cross-stage disjointness (full-session integration) ─────────────────────────
describe('cross-stage disjointness', () => {
    it('no image appears as a distinct-trial image in both stages', () => {
        const cfg = {
            experimental_trials: { images_per_trial: 5 },
            screening_block:     { enabled: true, num_experimental_trials: 2, num_repeat_trials: 1, min_repeat_separation: 1, num_catch_trials: 1 },
            experimental_block:  { num_experimental_trials: 2, num_repeat_trials: 1, min_repeat_separation: 1, num_catch_trials: 1 },
            catch_trials:        { images_per_trial: 4 },
        };
        const nNeeded = (cfg.screening_block.num_experimental_trials + cfg.experimental_block.num_experimental_trials) * cfg.experimental_trials.images_per_trial;
        const images = Array.from({ length: nNeeded }, (_, i) => `img_${i}.png`);
        const catchPool = Array.from({ length: 20 }, (_, i) => `catch_${i}.png`);

        const rng = makeRng(99);
        const distinctTrials = buildTrialLists(images, cfg, rng);
        const { screening: screeningDistinct, experimental: experimentalDistinct } = partitionIntoStages(distinctTrials, cfg);
        const screeningTrials = buildStage(screeningDistinct, catchPool, cfg.screening_block, cfg.catch_trials, rng, 'screening');
        const experimentalTrials = buildStage(experimentalDistinct, catchPool, cfg.experimental_block, cfg.catch_trials, rng, 'experimental');

        const imageToStages = {};
        [screeningTrials, experimentalTrials].forEach(stageTrials => {
            stageTrials.filter(t => t.type === 'main' && !t.isRepeat).forEach(t => {
                t.images.forEach(img => {
                    imageToStages[img] = imageToStages[img] || new Set();
                    imageToStages[img].add(t.block);
                });
            });
        });
        Object.values(imageToStages).forEach(stageSet => assert.equal(stageSet.size, 1));
    });
});
