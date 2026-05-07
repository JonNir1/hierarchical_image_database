'use strict';

// Requires: utils.js must be loaded before this file (uses seededShuffle).

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/**
 * Return the element of `array` for which `fn(element)` is smallest.
 * @param {Array} array
 * @param {function(*): number} fn
 * @returns {*} Element with minimum fn value
 */
function _minBy(array, fn) {
    let best    = array[0];
    let bestVal = fn(best);
    for (let i = 1; i < array.length; i++) {
        const val = fn(array[i]);
        if (val < bestVal) { best = array[i]; bestVal = val; }
    }
    return best;
}

/**
 * Return the indices of trials that have room and do not already contain `img`.
 * @param {string[][]} trials
 * @param {string}     img
 * @param {number}     k     - Max images per trial
 * @returns {number[]}
 */
function _eligibleIndices(trials, img, k) {
    const eligible = [];
    for (let i = 0; i < trials.length; i++) {
        if (trials[i].length < k && trials[i].indexOf(img) === -1) eligible.push(i);
    }
    return eligible;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Build per-subject trial lists with controlled image repetition.
 *
 * Each subject sees `n_unique` unique images across `t` trials of `k` images.
 * To allow within-subject reliability estimation, `n_double = t*k − n_unique`
 * images appear in exactly 2 trials each; the rest appear once.
 * No image ever appears more than once within a single trial.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{trials_per_subject: number,
 *           images_per_trial: number,
 *           unique_images_per_subject: number}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {string[][]} Array of `t` arrays, each containing exactly `k` image paths
 * @throws {Error} If assignment leaves any trial underfilled (indicates a config bug)
 */
function buildTrialLists(allImages, config, rng) {
    const t        = config.trials_per_subject;
    const k        = config.images_per_trial;
    const n_unique = config.unique_images_per_subject;
    const n_double = t * k - n_unique; // images that appear in 2 trials each

    // Subject-specific random subset of the full image pool
    const activeSet = seededShuffle(allImages, rng).slice(0, n_unique);

    // Partition: first n_double images appear twice, remainder appear once
    const doubleImages = activeSet.slice(0, n_double);
    const singleImages = seededShuffle(activeSet.slice(n_double), rng);

    // Initialise empty trial slots
    const trials = Array.from({ length: t }, () => []);

    // --- Pass 1: assign each double-image to exactly 2 distinct trials ---
    for (const img of doubleImages) {
        const eligible = _eligibleIndices(trials, img, k);
        if (eligible.length < 2) {
            throw new Error(
                `buildTrialLists: fewer than 2 eligible trials for double-image "${img}". ` +
                'Check trials_per_subject, images_per_trial, and unique_images_per_subject.'
            );
        }
        const shuffled = seededShuffle(eligible, rng);
        trials[shuffled[0]].push(img);
        trials[shuffled[1]].push(img);
    }

    // --- Pass 2: assign each single-image to the least-full eligible trial ---
    for (const img of singleImages) {
        const eligible = _eligibleIndices(trials, img, k);
        if (eligible.length === 0) continue; // underfill caught by validation below
        const bestIdx = _minBy(eligible, idx => trials[idx].length);
        trials[bestIdx].push(img);
    }

    // --- Shuffle image order within each trial ---
    for (let i = 0; i < t; i++) {
        trials[i] = seededShuffle(trials[i], rng);
    }

    // --- Validate ---
    for (let i = 0; i < t; i++) {
        if (trials[i].length < k) {
            throw new Error(
                `buildTrialLists: trial ${i} has ${trials[i].length} images, expected ${k}. ` +
                'Check config parameters.'
            );
        }
    }

    return trials;
}

