// Run: node --test SpAM_Task/__tests__/catch_trials.test.js
'use strict';

const { describe, it, before } = require('node:test');
const assert = require('node:assert/strict');
const { load, makeRng, ALL_IMAGES, CONFIG: BASE_CONFIG } = require('./helpers');

load('../js/utils.js');
load('../js/trial_generator.js');

const CATCH_POOL = Array.from({ length: 20 }, (_, i) => `catch_${i}.png`);
const IMAGES_PER_CATCH_TRIAL = 10;
const CONFIG = {
    ...BASE_CONFIG,
    screening_block:    { enabled: false, num_experimental_trials: 0 },
    experimental_block: { num_experimental_trials: 10 },
};
const SORT_W = 900;
const SORT_H = 700;

// ── buildCatchTrial ───────────────────────────────────────────────────────────
describe('buildCatchTrial', () => {
    it('returns a catch trial with exactly imagesPerTrial images and one of the five valid target_locations', () => {
        const trial = buildCatchTrial(CATCH_POOL, IMAGES_PER_CATCH_TRIAL, makeRng());
        assert.equal(trial.type, 'catch');
        assert.ok(Array.isArray(trial.images));
        assert.equal(trial.images.length, IMAGES_PER_CATCH_TRIAL);
        assert.ok(typeof trial.target_location === 'string');
        assert.ok(CATCH_LOCATIONS.includes(trial.target_location));
    });

    it('is deterministic with the same rng seed and differs across seeds', () => {
        const a = buildCatchTrial(CATCH_POOL, IMAGES_PER_CATCH_TRIAL, makeRng(99));
        const b = buildCatchTrial(CATCH_POOL, IMAGES_PER_CATCH_TRIAL, makeRng(99));
        assert.deepEqual(a, b);

        const c = buildCatchTrial(CATCH_POOL, IMAGES_PER_CATCH_TRIAL, makeRng(1));
        const d = buildCatchTrial(CATCH_POOL, IMAGES_PER_CATCH_TRIAL, makeRng(2));
        // Not guaranteed but overwhelmingly likely with any reasonable RNG
        assert.ok(
            c.target_location !== d.target_location || !c.images.every((x, i) => x === d.images[i]),
            'Expected different seeds to produce different trials'
        );
    });
});

// ── insertCatchTrials ─────────────────────────────────────────────────────────
describe('insertCatchTrials', () => {
    let mains, result;
    before(() => {
        mains  = buildTrialLists(ALL_IMAGES, CONFIG, makeRng());
        result = insertCatchTrials(mains, CATCH_POOL, 2, IMAGES_PER_CATCH_TRIAL, makeRng());
    });

    it('combined length is numMain + numCatch, with catch trials at positions 3 and 7 carrying a target_location', () => {
        assert.equal(result.length, 12);
        assert.equal(result[3].type, 'catch');
        assert.equal(result[7].type, 'catch');
        result.filter(t => t.type === 'catch').forEach(t => assert.ok(CATCH_LOCATIONS.includes(t.target_location)));
    });

    it('main trials are preserved in order', () => {
        const mainOut = result.filter(t => t.type === 'main');
        assert.equal(mainOut.length, 10);
        mainOut.forEach((t, i) => assert.deepEqual(t.images, mains[i]));
    });
});

// ── computeCentroid ───────────────────────────────────────────────────────────
describe('computeCentroid', () => {
    const cases = [
        ['returns the mean x and y', [{ src: 'a', x: 100, y: 200 }, { src: 'b', x: 300, y: 400 }], { x: 200, y: 300 }],
        ['handles a single image', [{ src: 'a', x: 50, y: 75 }], { x: 50, y: 75 }],
        ['returns {0,0} for empty array', [], { x: 0, y: 0 }],
    ];
    for (const [name, locs, expected] of cases) {
        it(name, () => {
            const c = computeCentroid(locs);
            assert.equal(c.x, expected.x);
            assert.equal(c.y, expected.y);
        });
    }
});

// ── allImagesNearTarget ───────────────────────────────────────────────────────
// allImagesNearTarget replaced isCentroidNearTarget in f372cde: every image
// must individually be within tolerance, not just the centroid.
describe('allImagesNearTarget', () => {
    const TOL = 0.20;

    // Helper: build n images all placed at (x, y).
    const cluster = (x, y, n = 3) =>
        Array.from({ length: n }, (_, i) => ({ src: `img_${i}.png`, x, y }));

    // Target points sit at EDGE=15% from each edge.
    const targets = {
        'center':              { x: SORT_W / 2,    y: SORT_H / 2 },
        'top left corner':     { x: SORT_W * 0.15, y: SORT_H * 0.15 },
        'top right corner':    { x: SORT_W * 0.85, y: SORT_H * 0.15 },
        'bottom left corner':  { x: SORT_W * 0.15, y: SORT_H * 0.85 },
        'bottom right corner': { x: SORT_W * 0.85, y: SORT_H * 0.85 },
    };

    for (const [target, point] of Object.entries(targets)) {
        it(`"${target}" passes when all images are at its own target point`, () => {
            assert.ok(allImagesNearTarget(cluster(point.x, point.y), target, SORT_W, SORT_H, TOL));
        });
    }

    it('"center" fails when all images are far from the centre', () => {
        assert.ok(!allImagesNearTarget(cluster(SORT_W * 0.9, SORT_H * 0.9), 'center', SORT_W, SORT_H, TOL));
    });

    for (const target of Object.keys(targets).filter(t => t !== 'center')) {
        it(`"${target}" fails when all images are at the centre`, () => {
            assert.ok(!allImagesNearTarget(cluster(SORT_W / 2, SORT_H / 2), target, SORT_W, SORT_H, TOL));
        });
    }

    it('one stray image causes failure even if others are on target', () => {
        const locs = cluster(SORT_W * 0.15, SORT_H * 0.15);
        locs.push({ src: 'stray.png', x: SORT_W / 2, y: SORT_H / 2 }); // far from corner
        assert.ok(!allImagesNearTarget(locs, 'top left corner', SORT_W, SORT_H, TOL));
    });

    it('returns false for empty locations array', () => {
        assert.ok(!allImagesNearTarget([], 'center', SORT_W, SORT_H, TOL));
    });
});
