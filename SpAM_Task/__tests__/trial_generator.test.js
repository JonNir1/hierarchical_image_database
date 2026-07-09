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
    design: {
        num_blocks:                  2,
        trials_per_block:            5,
        images_per_trial:            20,
        repeats_per_block:           0,
        min_trial_repeat_separation: 2,
    },
    catch_trials: {
        catch_per_block:   1,
        images_per_trial:  10,
    },
};

// ── buildTrialLists ───────────────────────────────────────────────────────────
describe('buildTrialLists', () => {
    it('returns num_blocks*trials_per_block trials each with exactly images_per_trial images', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        assert.equal(trials.length, 10);
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
        // 2 images, num_blocks=1 * trials_per_block=3 * images_per_trial=2 = 6 needed — impossible
        assert.throws(() =>
            buildTrialLists(['x.png','y.png'],
                { design: { num_blocks: 1, trials_per_block: 3, images_per_trial: 2 } },
                makeRng()),
            { message: /image pool has 2 image\(s\), need 6/ }
        );
    });
    it('returns num_blocks*trials_per_block trials regardless of repeats_per_block (additive, not substitutive)', () => {
        const cfg = { design: { ...CONFIG.design, repeats_per_block: 2 } };
        const trials = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.equal(trials.length, 10); // unchanged — repeats are added on top, not carved out
    });
});

// ── partitionIntoBlocks ────────────────────────────────────────────────────────
describe('partitionIntoBlocks', () => {
    it('splits into num_blocks consecutive groups of trials_per_block, order preserved, no overlap', () => {
        const cfg = { design: { num_blocks: 3, trials_per_block: 4 } };
        const distinctTrials = Array.from({ length: 12 }, (_, i) => [`t${i}_a`, `t${i}_b`]);
        const blocks = partitionIntoBlocks(distinctTrials, cfg);
        assert.equal(blocks.length, 3);
        blocks.forEach(b => assert.equal(b.length, 4));
        assert.deepEqual(blocks[0], distinctTrials.slice(0, 4));
        assert.deepEqual(blocks[1], distinctTrials.slice(4, 8));
        assert.deepEqual(blocks[2], distinctTrials.slice(8, 12));
    });
});

// ── insertTrialRepeats ────────────────────────────────────────────────────────
describe('insertTrialRepeats', () => {
    it('returns trials unchanged (wrapped) when repeats_per_block is 0', () => {
        const trials = buildTrialLists(ALL_IMAGES, CONFIG, makeRng()).slice(0, 5);
        const result = insertTrialRepeats(trials, CONFIG, makeRng(), 0);
        assert.equal(result.length, 5);
        result.forEach((r, i) => {
            assert.equal(r.isRepeat, false);
            assert.equal(r.repeatOfTrialId, null);
            assert.equal(r.trialId, '0_' + i);
            assert.deepEqual(r.images, trials[i]);
        });
    });

    it('produces exactly repeats_per_block repeat entries with matching image sets', () => {
        const cfg = { design: { ...CONFIG.design, repeats_per_block: 3 } }; // block-local: 5 distinct + 3 repeats
        const trials = buildTrialLists(ALL_IMAGES, cfg, makeRng()).slice(0, 5);
        const result = insertTrialRepeats(trials, cfg, makeRng(), 0);

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
        const cfg = { design: { trials_per_block: 4, repeats_per_block: 1, min_trial_repeat_separation: 1 } };
        const result = insertTrialRepeats(trials, cfg, makeRng(), 0);
        assert.equal(result.filter(r => r.isRepeat).length, 1);
    });

    it('respects min_trial_repeat_separation (block-local)', () => {
        // trials_per_block=7 + repeats_per_block=3 -> tBlock=10, minSep=3: known-feasible layout.
        const cfg = { design: { ...CONFIG.design, trials_per_block: 7, repeats_per_block: 3, min_trial_repeat_separation: 3 } };
        const trials = buildTrialLists(ALL_IMAGES, { design: { num_blocks: 1, trials_per_block: 7, images_per_trial: cfg.design.images_per_trial } }, makeRng());
        const result = insertTrialRepeats(trials, cfg, makeRng(), 0);
        const positionOf = id => result.findIndex(r => r.trialId === id);
        result.forEach((rep, pos) => {
            if (rep.isRepeat) {
                assert.ok(pos - positionOf(rep.repeatOfTrialId) >= 3);
            }
        });
    });

    it('is deterministic with the same seed', () => {
        const cfg = { design: { ...CONFIG.design, repeats_per_block: 3 } };
        const run = seed => {
            const rng = makeRng(seed);
            const trials = buildTrialLists(ALL_IMAGES, cfg, rng).slice(0, 5);
            return insertTrialRepeats(trials, cfg, rng, 0);
        };
        assert.deepEqual(run(7), run(7));
    });

    it('namespaces trialId by blockIndex, disjoint across blocks', () => {
        const trials = [['a','b'], ['c','d']];
        const cfg = { design: { trials_per_block: 2, repeats_per_block: 0, min_trial_repeat_separation: 1 } };
        const resultBlock0 = insertTrialRepeats(trials, cfg, makeRng(1), 0);
        const resultBlock1 = insertTrialRepeats(trials, cfg, makeRng(1), 1);
        assert.deepEqual(resultBlock0.map(r => r.trialId), ['0_0', '0_1']);
        assert.deepEqual(resultBlock1.map(r => r.trialId), ['1_0', '1_1']);
    });

    it('throws when min_trial_repeat_separation cannot be satisfied within a block', () => {
        const cfg = {
            design: {
                trials_per_block: 2, images_per_trial: 20,
                repeats_per_block: 2, // block-local t = 2+2 = 4
                min_trial_repeat_separation: 3, // impossible to fit 2 repeats with gap 3 in 4 slots
            },
        };
        const trials = buildTrialLists(ALL_IMAGES, { design: { num_blocks: 1, ...cfg.design } }, makeRng());
        assert.throws(
            () => insertTrialRepeats(trials, cfg, makeRng(), 0),
            { message: /min_trial_repeat_separation/ }
        );
    });
});

// ── buildBlock ────────────────────────────────────────────────────────────────
describe('buildBlock', () => {
    it('returns trials_per_block + repeats_per_block + catch_per_block trials, all stamped with block', () => {
        const cfg = { design: { ...CONFIG.design, repeats_per_block: 1 }, catch_trials: CONFIG.catch_trials };
        const distinctTrials = buildTrialLists(ALL_IMAGES, { design: { num_blocks: 1, trials_per_block: cfg.design.trials_per_block, images_per_trial: cfg.design.images_per_trial } }, makeRng());
        const catchPool = Array.from({ length: 50 }, (_, i) => `catch_${i}.png`);
        const combined = buildBlock(distinctTrials, catchPool, cfg, makeRng(), 2); // blockIndex 2 -> block 3

        assert.equal(combined.length, 5 + 1 + 1); // trials_per_block + repeats_per_block + catch_per_block
        combined.forEach(t => assert.equal(t.block, 3));

        const repeat = combined.find(t => t.type === 'main' && t.isRepeat);
        const original = combined.find(t => t.type === 'main' && t.trialId === repeat.repeatOfTrialId);
        assert.deepEqual([...repeat.images].sort(), [...original.images].sort());
    });
});

// ── Cross-block disjointness (full-session integration) ────────────────────────
describe('cross-block disjointness', () => {
    it('no image appears as a distinct-trial image in two different blocks', () => {
        const cfg = {
            design: { num_blocks: 3, trials_per_block: 2, images_per_trial: 5, repeats_per_block: 1, min_trial_repeat_separation: 1 },
            catch_trials: { catch_per_block: 1, images_per_trial: 4 },
        };
        const nNeeded = cfg.design.num_blocks * cfg.design.trials_per_block * cfg.design.images_per_trial;
        const images = Array.from({ length: nNeeded }, (_, i) => `img_${i}.png`);
        const catchPool = Array.from({ length: 20 }, (_, i) => `catch_${i}.png`);

        const rng = makeRng(99);
        const distinctTrials   = buildTrialLists(images, cfg, rng);
        const blocksOfDistinct = partitionIntoBlocks(distinctTrials, cfg);
        const blocks = blocksOfDistinct.map((bt, i) => buildBlock(bt, catchPool, cfg, rng, i));

        const imageToBlocks = {};
        blocks.forEach((blockTrials, blockIdx) => {
            blockTrials.filter(t => t.type === 'main' && !t.isRepeat).forEach(t => {
                t.images.forEach(img => {
                    imageToBlocks[img] = imageToBlocks[img] || new Set();
                    imageToBlocks[img].add(blockIdx);
                });
            });
        });
        Object.values(imageToBlocks).forEach(blockSet => assert.equal(blockSet.size, 1));
    });
});
