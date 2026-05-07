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
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(900, 700, 900, 700));
        assert.equal(sortW,    900);
        assert.equal(sortH,    700);
        assert.equal(stimSize,  99); // round(900 * 0.11)
    });

    it('just-fits width — viewport slightly above min still yields min', () => {
        // floor(980 * 0.92) = 901 > max 900  →  capped at 900
        // floor(768 * 0.85) = 652 < min 700  →  floored at 700
        const { sortW, sortH } = computeLayout(980, 768, cfg(900, 700, 900, 700));
        assert.equal(sortW, 900);
        assert.equal(sortH, 700);
    });

    it('small screen with default mins — floor holds at 900×700', () => {
        // 800×600: viewport-derived values are below min → floor wins
        const { sortW, sortH, stimSize } = computeLayout(800, 600, cfg(900, 700, 900, 700));
        assert.equal(sortW,    900);
        assert.equal(sortH,    700);
        assert.equal(stimSize,  99);
    });

    it('small screen with low mins — shrinks to viewport-derived size', () => {
        // floor(700 * 0.92) = 644;  floor(500 * 0.85) = 425
        // both above min (400×350) and below max (900×700)  →  viewport wins
        const { sortW, sortH, stimSize } = computeLayout(700, 500, cfg(900, 700, 400, 350));
        assert.equal(sortW,    644);
        assert.equal(sortH,    425);
        assert.equal(stimSize,  71); // round(644 * 0.11)
    });

    it('large screen with high max — caps at the higher configured max', () => {
        // floor(1920 * 0.92) = 1766 > max 1100  →  capped at 1100
        // floor(1080 * 0.85) = 918  > max 800   →  capped at 800
        const { sortW, sortH, stimSize } = computeLayout(1920, 1080, cfg(1100, 800, 600, 500));
        assert.equal(sortW,    1100);
        assert.equal(sortH,    800);
        assert.equal(stimSize, 121); // round(1100 * 0.11)
    });
});
