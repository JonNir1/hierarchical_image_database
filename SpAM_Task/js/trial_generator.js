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
 * Number of genuinely distinct trial combinations to generate, after setting
 * aside slots that will instead hold verbatim repeats of earlier trials
 * (see insertTrialRepeats).
 * @param {{design: {trials_per_subject: number, frac_trials_repeated: number}}} config
 * @returns {number}
 */
function _distinctTrialCount(config) {
    const t  = config.design.trials_per_subject;
    const fr = config.design.frac_trials_repeated || 0;
    return t - Math.round(fr * t);
}

/**
 * Build per-subject trial lists with controlled image repetition.
 *
 * Each subject sees n_unique = round(t_distinct*k / (1 + r)) unique images across
 * t_distinct distinct trials of k images, where r = frac_images_repeated and
 * t_distinct = trials_per_subject - round(frac_trials_repeated * trials_per_subject).
 * To allow within-subject reliability estimation, n_double = round(r * n_unique)
 * images appear in exactly 2 trials each; the rest appear once.
 * No image ever appears more than once within a single trial.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{design: {trials_per_subject: number,
 *                   images_per_trial: number,
 *                   frac_images_repeated: number,
 *                   frac_trials_repeated: number}}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {{trials: string[][], doubleImages: Set<string>}} `t_distinct` trials of
 *          exactly `k` images each, plus the set of images appearing in 2 of them
 * @throws {Error} If assignment leaves any trial underfilled (indicates a config bug)
 */
function buildTrialLists(allImages, config, rng) {
    const t        = _distinctTrialCount(config);
    const k        = config.design.images_per_trial;
    const r        = config.design.frac_images_repeated; // keep < 0.5; greedy placement can fail above that
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
                'Check trials_per_subject, images_per_trial, and frac_images_repeated.'
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

    return { trials, doubleImages: new Set(doubleImages) };
}

// ---------------------------------------------------------------------------
// Trial repeats
// ---------------------------------------------------------------------------

/**
 * Fill the remaining `trials_per_subject - trials.length` slots with verbatim
 * repeats of earlier trials, for test-retest reliability of the arrangement
 * response itself (distinct from the cross-context reliability measured by
 * frac_images_repeated).
 *
 * Repeats are built only from "singles-only" trials (no image in
 * `doubleImages`), so no image can end up appearing in more than 2 trials
 * total via the combination of the two mechanisms. Each repeat is placed at
 * least `min_trial_repeat_separation` slots after the trial it duplicates
 * (image order is reshuffled so the repeat isn't pixel-identical to the
 * original, though the image set is).
 *
 * @param {string[][]} trials - Output of buildTrialLists (`trials` field), length t_distinct
 * @param {Set<string>} doubleImages - Output of buildTrialLists (`doubleImages` field)
 * @param {{design: {trials_per_subject: number, min_trial_repeat_separation: number}}} config
 * @param {function(): number} rng - Seeded RNG (same instance used throughout)
 * @returns {Array<{images: string[], isRepeat: boolean, repeatOfTrialId: (number|null), trialId: (number|string)}>}
 *          Array of length `trials_per_subject`
 * @throws {Error} If there aren't enough singles-only trials, or no placement
 *                 satisfies min_trial_repeat_separation
 */
function insertTrialRepeats(trials, doubleImages, config, rng) {
    const t           = config.design.trials_per_subject;
    const numRepeats  = t - trials.length;
    const minSep      = config.design.min_trial_repeat_separation;

    if (numRepeats === 0) {
        return trials.map((images, i) => ({ images, isRepeat: false, repeatOfTrialId: null, trialId: i }));
    }

    const isSinglesOnly = idx => trials[idx].every(img => !doubleImages.has(img));
    const candidates     = trials.map((_, i) => i).filter(isSinglesOnly);
    if (candidates.length < numRepeats) {
        throw new Error(
            `insertTrialRepeats: only ${candidates.length} singles-only trial(s) available, ` +
            `need ${numRepeats} for the configured frac_trials_repeated. Lower ` +
            'frac_trials_repeated or frac_images_repeated.'
        );
    }
    // Structural layout (which of the t slots are repeats vs. distinct
    // trials) is shared with verifyConfig's static feasibility pre-check —
    // see computeRepeatLayout in utils.js.
    const { repeatPositions, distinctPositions } = computeRepeatLayout(t, numRepeats);

    const sequence         = new Array(t).fill(null);
    const originalPosition = {};
    distinctPositions.forEach((pos, distinctIdx) => {
        sequence[pos] = { images: trials[distinctIdx], isRepeat: false, repeatOfTrialId: null, trialId: distinctIdx };
        originalPosition[distinctIdx] = pos;
    });

    // Assign candidate originals to repeat slots in increasing position order
    // (most-constrained first), enforcing min_trial_repeat_separation. Drawing
    // from the full candidate pool here (rather than a pre-selected subset)
    // maximises the chance an eligible original exists for every slot.
    const remaining = seededShuffle(candidates, rng);
    for (const pos of repeatPositions) {
        const eligibleIdx = remaining.findIndex(id => pos - originalPosition[id] >= minSep);
        if (eligibleIdx === -1) {
            throw new Error(
                `insertTrialRepeats: cannot place a repeat at position ${pos} satisfying ` +
                `min_trial_repeat_separation=${minSep}. Reduce frac_trials_repeated, lower ` +
                'min_trial_repeat_separation, or increase trials_per_subject.'
            );
        }
        const id = remaining[eligibleIdx];
        remaining.splice(eligibleIdx, 1);
        sequence[pos] = {
            images:          seededShuffle(trials[id], rng),
            isRepeat:        true,
            repeatOfTrialId: id,
            trialId:         id + '_repeat',
        };
    }

    return sequence;
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
 * @param {Array<{images: string[], isRepeat?: boolean, repeatOfTrialId?: (number|string|null), trialId?: (number|string)}>} mainTrials
 *        Output of insertTrialRepeats (or plain `string[][]`, for backward compatibility)
 * @param {string[]}           catchPool  - Images reserved for catch trials
 * @param {{catch_trials: {num_trials: number, images_per_trial: number}}} config
 * @param {function(): number} rng        - Seeded RNG (same instance used throughout)
 * @returns {Array<{type: 'main'|'catch', images: string[], target_location?: string, isRepeat?: boolean, repeatOfTrialId?: (number|string|null), trialId?: (number|string)}>}
 */
function insertCatchTrials(mainTrials, catchPool, config, rng) {
    const numMain  = mainTrials.length;
    const numCatch = config.catch_trials.num_trials;

    const catchPositions = [];
    for (let i = 1; i <= numCatch; i++) {
        catchPositions.push(Math.round(numMain / (numCatch + 1) * i));
    }

    const mainObjects = mainTrials.map(m =>
        Array.isArray(m) ? { type: 'main', images: m } : { type: 'main', ...m }
    );
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
