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
const CONFIG = {
    design: {
        trials_per_subject:          10,
        images_per_trial:            20,
        frac_images_repeated:        1/3,
        frac_trials_repeated:        0,
        min_trial_repeat_separation: 2,
    },
};

// ── buildTrialLists ───────────────────────────────────────────────────────────
describe('buildTrialLists', () => {
    it('returns t_distinct trials each with exactly k images', () => {
        const { trials } = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        assert.equal(trials.length, 10);
        trials.forEach(t => assert.equal(t.length, 20));
    });
    it('no duplicates within a single trial', () => {
        buildTrialLists(ALL_IMAGES, CONFIG, makeRng()).trials
            .forEach(t => assert.equal(new Set(t).size, t.length));
    });
    it('exactly n_double images appear in 2 trials; the rest appear in 1', () => {
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        const counts = {};
        trials.flat().forEach(img => { counts[img] = (counts[img] || 0) + 1; });
        const freq = Object.values(counts);
        assert.equal(freq.filter(c => c === 2).length, 50);
        assert.ok(freq.every(c => c === 1 || c === 2));
        assert.equal(doubleImages.size, 50);
    });
    it('is deterministic with the same seed', () => {
        assert.deepEqual(
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7)).trials,
            buildTrialLists(ALL_IMAGES, CONFIG, makeRng(7)).trials
        );
    });
    it('throws when the image pool cannot fill all trials', () => {
        // 2 images, 3 trials × 2 slots — impossible to fill every trial
        // r=0 → N = round(6/1) = 6, but pool only has 2 images
        assert.throws(() =>
            buildTrialLists(['x.png','y.png'],
                { design: { trials_per_subject:3, images_per_trial:2, frac_images_repeated:0, frac_trials_repeated:0 } },
                makeRng())
        );
    });
    it('builds fewer distinct trials when frac_trials_repeated > 0', () => {
        const cfg = { design: { ...CONFIG.design, frac_trials_repeated: 0.3 } }; // t_distinct = 10 - round(3) = 7
        const { trials } = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.equal(trials.length, 7);
    });
});

// ── insertTrialRepeats ────────────────────────────────────────────────────────
describe('insertTrialRepeats', () => {
    it('returns trials unchanged (wrapped) when frac_trials_repeated is 0', () => {
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        const result = insertTrialRepeats(trials, doubleImages, CONFIG, makeRng());
        assert.equal(result.length, 10);
        result.forEach((r, i) => {
            assert.equal(r.isRepeat, false);
            assert.equal(r.repeatOfTrialId, null);
            assert.deepEqual(r.images, trials[i]);
        });
    });

    // frac_images_repeated is 0 in these cases so every distinct trial is
    // singles-only and unconstrained by the cross-bucket rule (tested separately below).
    it('produces exactly n_trial_repeats repeat entries with matching image sets', () => {
        const cfg = { design: { ...CONFIG.design, frac_images_repeated: 0, frac_trials_repeated: 0.3 } }; // t_distinct=7, numRepeats=3
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        const result = insertTrialRepeats(trials, doubleImages, cfg, makeRng());

        assert.equal(result.length, 10);
        const repeats = result.filter(r => r.isRepeat);
        assert.equal(repeats.length, 3);
        repeats.forEach(rep => {
            const original = result.find(r => r.trialId === rep.repeatOfTrialId);
            assert.ok(original);
            assert.deepEqual([...rep.images].sort(), [...original.images].sort());
        });
    });

    it('never repeats a trial containing a double image', () => {
        // Manually constructed fixture (not RNG-derived) so the singles-only
        // constraint is checked against a known, controlled trial composition:
        // trials 0 and 3 each contain the double image 'a' and must never be
        // chosen for repetition; trials 1 and 2 are singles-only candidates.
        const trials       = [['a','b','c'], ['d','e','f'], ['g','h','i'], ['j','k','a']];
        const doubleImages = new Set(['a']);
        const cfg = { design: { trials_per_subject: 5, frac_trials_repeated: 0.2, min_trial_repeat_separation: 1 } }; // t_distinct=4, numRepeats=1
        const result = insertTrialRepeats(trials, doubleImages, cfg, makeRng());
        assert.equal(result.filter(r => r.isRepeat).length, 1);
        result.filter(r => r.isRepeat).forEach(rep => {
            assert.ok(rep.repeatOfTrialId === 1 || rep.repeatOfTrialId === 2);
            rep.images.forEach(img => assert.ok(!doubleImages.has(img)));
        });
    });

    it('respects min_trial_repeat_separation', () => {
        const cfg = { design: { ...CONFIG.design, frac_images_repeated: 0, frac_trials_repeated: 0.3, min_trial_repeat_separation: 3 } };
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        const result = insertTrialRepeats(trials, doubleImages, cfg, makeRng());
        const positionOf = id => result.findIndex(r => r.trialId === id);
        result.forEach((rep, pos) => {
            if (rep.isRepeat) {
                assert.ok(pos - positionOf(rep.repeatOfTrialId) >= 3);
            }
        });
    });

    it('is deterministic with the same seed', () => {
        const cfg = { design: { ...CONFIG.design, frac_images_repeated: 0, frac_trials_repeated: 0.3 } };
        const run = seed => {
            const rng = makeRng(seed);
            const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, cfg, rng);
            return insertTrialRepeats(trials, doubleImages, cfg, rng);
        };
        assert.deepEqual(run(7), run(7));
    });

    it('throws when there are not enough singles-only trials for the requested repeats', () => {
        // r close to 1 (just under the hard [0,1) bound) forces nearly every
        // trial to contain at least one double image, leaving too few
        // singles-only candidates for frac_trials_repeated to draw from.
        const cfg = {
            design: {
                trials_per_subject: 10, images_per_trial: 20,
                frac_images_repeated: 0.49, frac_trials_repeated: 0.4,
                min_trial_repeat_separation: 2,
            },
        };
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.throws(
            () => insertTrialRepeats(trials, doubleImages, cfg, makeRng()),
            { message: /singles-only trial/ }
        );
    });

    it('throws when min_trial_repeat_separation cannot be satisfied', () => {
        const cfg = {
            design: {
                trials_per_subject: 4, images_per_trial: 20,
                frac_images_repeated: 0, frac_trials_repeated: 0.5, // t_distinct=2, numRepeats=2
                min_trial_repeat_separation: 3, // impossible to fit 2 repeats with gap 3 in 4 slots
            },
        };
        const { trials, doubleImages } = buildTrialLists(ALL_IMAGES, cfg, makeRng());
        assert.throws(
            () => insertTrialRepeats(trials, doubleImages, cfg, makeRng()),
            { message: /min_trial_repeat_separation/ }
        );
    });
});
