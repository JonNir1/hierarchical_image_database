// Run: node --test SpAM_Task/__tests__/catch_trials.test.js
'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const vm     = require('node:vm');
const fs     = require('node:fs');
const path   = require('node:path');

const load = f => vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, f), 'utf8'));
load('../js/utils.js');
load('../js/trial_generator.js');

// Deterministic LCG used across all tests (avoids seedrandom dependency in Node).
const makeRng = (seed = 42) => {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
};

const CATCH_POOL = Array.from({ length: 20 }, (_, i) => `catch_${i}.png`);
const CONFIG = {
    design: {
        trials_per_subject:        10,
        images_per_trial:          20,
        unique_images_per_subject: 150,
    },
    catch_trials: {
        num_trials:       2,
        images_per_trial: 10,
    },
};
const SORT_W = 900;
const SORT_H = 700;

// ── buildCatchTrial ───────────────────────────────────────────────────────────
describe('buildCatchTrial', () => {
    it('returns an object with type catch, images, and target_location', () => {
        const trial = buildCatchTrial(CATCH_POOL, CONFIG, makeRng());
        assert.equal(trial.type, 'catch');
        assert.ok(Array.isArray(trial.images));
        assert.ok(typeof trial.target_location === 'string');
    });

    it('samples exactly catch_images_per_trial images', () => {
        const trial = buildCatchTrial(CATCH_POOL, CONFIG, makeRng());
        assert.equal(trial.images.length, CONFIG.catch_trials.images_per_trial);
    });

    it('target_location is one of the five valid options', () => {
        const trial = buildCatchTrial(CATCH_POOL, CONFIG, makeRng());
        assert.ok(CATCH_LOCATIONS.includes(trial.target_location));
    });

    it('is deterministic with the same rng seed', () => {
        const a = buildCatchTrial(CATCH_POOL, CONFIG, makeRng(99));
        const b = buildCatchTrial(CATCH_POOL, CONFIG, makeRng(99));
        assert.deepEqual(a, b);
    });

    it('produces different results with different seeds', () => {
        const a = buildCatchTrial(CATCH_POOL, CONFIG, makeRng(1));
        const b = buildCatchTrial(CATCH_POOL, CONFIG, makeRng(2));
        // Not guaranteed but overwhelmingly likely with any reasonable RNG
        assert.ok(
            a.target_location !== b.target_location || !a.images.every((x, i) => x === b.images[i]),
            'Expected different seeds to produce different trials'
        );
    });
});

// ── insertCatchTrials ─────────────────────────────────────────────────────────
describe('insertCatchTrials', () => {
    const ALL_IMAGES = Array.from({ length: 754 }, (_, i) => `img_${i}.png`);
    const freshMains = () => buildTrialLists(ALL_IMAGES, CONFIG, makeRng());

    it('combined length is numMain + numCatch', () => {
        const result = insertCatchTrials(freshMains(), CATCH_POOL, CONFIG, makeRng());
        assert.equal(result.length, 12);
    });

    it('catch trials land at positions 3 and 7', () => {
        const result = insertCatchTrials(freshMains(), CATCH_POOL, CONFIG, makeRng());
        assert.equal(result[3].type, 'catch');
        assert.equal(result[7].type, 'catch');
    });

    it('catch trials carry a target_location string', () => {
        const result = insertCatchTrials(freshMains(), CATCH_POOL, CONFIG, makeRng());
        const catches = result.filter(t => t.type === 'catch');
        catches.forEach(t => assert.ok(CATCH_LOCATIONS.includes(t.target_location)));
    });

    it('main trials are preserved in order', () => {
        const mains  = freshMains();
        const result = insertCatchTrials(mains, CATCH_POOL, CONFIG, makeRng());
        const mainOut = result.filter(t => t.type === 'main');
        assert.equal(mainOut.length, 10);
        mainOut.forEach((t, i) => assert.deepEqual(t.images, mains[i]));
    });
});

// ── computeCentroid ───────────────────────────────────────────────────────────
describe('computeCentroid', () => {
    it('returns the mean x and y', () => {
        const locs = [
            { src: 'a', x: 100, y: 200 },
            { src: 'b', x: 300, y: 400 },
        ];
        const c = computeCentroid(locs);
        assert.equal(c.x, 200);
        assert.equal(c.y, 300);
    });

    it('handles a single image', () => {
        const c = computeCentroid([{ src: 'a', x: 50, y: 75 }]);
        assert.equal(c.x, 50);
        assert.equal(c.y, 75);
    });

    it('returns {0,0} for empty array', () => {
        const c = computeCentroid([]);
        assert.equal(c.x, 0);
        assert.equal(c.y, 0);
    });
});

// ── isCentroidNearTarget ──────────────────────────────────────────────────────
describe('isCentroidNearTarget', () => {
    const TOL = 0.20;

    it('centroid at exact centre passes "center"', () => {
        const c = { x: SORT_W / 2, y: SORT_H / 2 };
        assert.ok(isCentroidNearTarget(c, 'center', SORT_W, SORT_H, TOL));
    });

    it('centroid far from centre fails "center"', () => {
        const c = { x: SORT_W * 0.9, y: SORT_H * 0.9 };
        assert.ok(!isCentroidNearTarget(c, 'center', SORT_W, SORT_H, TOL));
    });

    it('centroid at exact top-left target passes "top left corner"', () => {
        // Target point is at EDGE=15% from each edge
        const c = { x: SORT_W * 0.15, y: SORT_H * 0.15 };
        assert.ok(isCentroidNearTarget(c, 'top left corner', SORT_W, SORT_H, TOL));
    });

    it('centroid at centre fails "top left corner"', () => {
        const c = { x: SORT_W / 2, y: SORT_H / 2 };
        assert.ok(!isCentroidNearTarget(c, 'top left corner', SORT_W, SORT_H, TOL));
    });

    it('each corner target passes only when centroid is near that corner', () => {
        const corners = {
            'top left corner':     { x: SORT_W * 0.15, y: SORT_H * 0.15 },
            'top right corner':    { x: SORT_W * 0.85, y: SORT_H * 0.15 },
            'bottom left corner':  { x: SORT_W * 0.15, y: SORT_H * 0.85 },
            'bottom right corner': { x: SORT_W * 0.85, y: SORT_H * 0.85 },
        };
        for (const [loc, pos] of Object.entries(corners)) {
            // Correct corner → pass
            assert.ok(
                isCentroidNearTarget(pos, loc, SORT_W, SORT_H, TOL),
                `Expected pass for ${loc} at its own target point`
            );
            // Centre → fail
            assert.ok(
                !isCentroidNearTarget({ x: SORT_W / 2, y: SORT_H / 2 }, loc, SORT_W, SORT_H, TOL),
                `Expected fail for ${loc} when centroid is at centre`
            );
        }
    });
});
