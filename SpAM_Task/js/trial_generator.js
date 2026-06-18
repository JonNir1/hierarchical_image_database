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
 * Each subject sees n_unique = round(t*k / (1 + r)) unique images across
 * t trials of k images, where r = percent_images_repeated.
 * To allow within-subject reliability estimation, n_double = round(r * n_unique)
 * images appear in exactly 2 trials each; the rest appear once.
 * No image ever appears more than once within a single trial.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{design: {trials_per_subject: number,
 *                   images_per_trial: number,
 *                   percent_images_repeated: number}}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {string[][]} Array of `t` arrays, each containing exactly `k` image paths
 * @throws {Error} If assignment leaves any trial underfilled (indicates a config bug)
 */
function buildTrialLists(allImages, config, rng) {
    const t        = config.design.trials_per_subject;
    const k        = config.design.images_per_trial;
    const r        = config.design.percent_images_repeated;
    const n_unique = Math.round(t * k / (1 + r));
    const n_double = t * k - n_unique; // = round(r * n_unique)

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
                'Check trials_per_subject, images_per_trial, and percent_images_repeated.'
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

// ---------------------------------------------------------------------------
// Catch trials
// ---------------------------------------------------------------------------

/**
 * The five target locations shown to participants in catch trials.
 * Exported as a constant so task.js and tests can reference the same list.
 */
const CATCH_LOCATIONS = [
    'center',
    'top left corner',
    'top right corner',
    'bottom left corner',
    'bottom right corner',
];

/**
 * Build a single catch trial.
 *
 * Samples `config.catch_images_per_trial` images from the catch pool and
 * draws a target location from CATCH_LOCATIONS — both via the shared seeded
 * RNG so the assignment is reproducible per participant.
 *
 * @param {string[]}          catchPool - Pool of images reserved for catch trials
 * @param {{catch_trials: {images_per_trial: number}}} config
 * @param {function(): number} rng      - Seeded RNG (shared with buildTrialLists)
 * @returns {{type: 'catch', images: string[], target_location: string}}
 */
function buildCatchTrial(catchPool, config, rng) {
    const k        = config.catch_trials.images_per_trial;
    const images   = seededShuffle(catchPool, rng).slice(0, k);
    const locIdx   = Math.floor(rng() * CATCH_LOCATIONS.length);
    return { type: 'catch', images, target_location: CATCH_LOCATIONS[locIdx] };
}

/**
 * Interleave catch trials at evenly spaced interior positions in the trial sequence.
 *
 * Insertion positions (in the combined array) are:
 *   Math.round(numMain / (numCatch + 1) * i)  for i = 1 … numCatch
 * Example: numMain=10, numCatch=2 → positions [3, 7].
 *
 * @param {string[][]}         mainTrials - Output of buildTrialLists
 * @param {string[]}           catchPool  - Images reserved for catch trials
 * @param {{catch_trials: {num_trials: number, images_per_trial: number}}} config
 * @param {function(): number} rng        - Seeded RNG (same instance used throughout)
 * @returns {Array<{type: 'main'|'catch', images: string[], target_location?: string}>}
 */
function insertCatchTrials(mainTrials, catchPool, config, rng) {
    const numMain  = mainTrials.length;
    const numCatch = config.catch_trials.num_trials;

    const catchPositions = [];
    for (let i = 1; i <= numCatch; i++) {
        catchPositions.push(Math.round(numMain / (numCatch + 1) * i));
    }

    const mainObjects = mainTrials.map(images => ({ type: 'main', images }));
    const combined    = [];
    let mainIdx  = 0;
    let catchIdx = 0;

    for (let pos = 0; pos < numMain + numCatch; pos++) {
        if (catchIdx < catchPositions.length && pos === catchPositions[catchIdx]) {
            combined.push(buildCatchTrial(catchPool, config, rng));
            catchIdx++;
        } else {
            combined.push(mainObjects[mainIdx++]);
        }
    }

    return combined;
}
