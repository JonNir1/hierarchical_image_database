'use strict';

// Requires: utils.js must be loaded before this file (uses seededShuffle).

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
 * Build per-subject trial lists with each image appearing in exactly one
 * distinct trial.
 *
 * Each subject sees t_distinct * k unique images across t_distinct distinct
 * trials of k images each, where t_distinct = trials_per_subject -
 * round(frac_trials_repeated * trials_per_subject). The only way an image
 * can ever appear more than once per subject is via a verbatim whole-trial
 * repeat (see insertTrialRepeats), independent of this function.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{design: {trials_per_subject: number,
 *                   images_per_trial: number,
 *                   frac_trials_repeated: number}}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {string[][]} `t_distinct` trials of exactly `k` images each
 * @throws {Error} If the image pool is too small to fill `t_distinct * k` slots
 */
function buildTrialLists(allImages, config, rng) {
    const t_distinct = _distinctTrialCount(config);
    const k          = config.design.images_per_trial;
    const n_needed   = t_distinct * k;

    if (allImages.length < n_needed) {
        throw new Error(
            `buildTrialLists: image pool has ${allImages.length} image(s), need ${n_needed} ` +
            `(trials_per_subject × images_per_trial, accounting for frac_trials_repeated). ` +
            'Add more images, or lower trials_per_subject/images_per_trial.'
        );
    }

    // Subject-specific random subset, sliced into t_distinct trials of k images each.
    const activeSet = seededShuffle(allImages, rng).slice(0, n_needed);
    const trials    = [];
    for (let i = 0; i < t_distinct; i++) {
        trials.push(activeSet.slice(i * k, (i + 1) * k));
    }

    return trials;
}

// ---------------------------------------------------------------------------
// Trial repeats
// ---------------------------------------------------------------------------

/**
 * Fill the remaining `trials_per_subject - trials.length` slots with verbatim
 * repeats of earlier trials, for test-retest reliability of the arrangement
 * response itself.
 *
 * Every distinct trial is eligible to be repeated (each image already
 * appears in exactly one distinct trial, by construction of buildTrialLists).
 * Each repeat is placed at least `min_trial_repeat_separation` slots after
 * the trial it duplicates (image order is reshuffled so the repeat isn't
 * pixel-identical to the original, though the image set is).
 *
 * @param {string[][]} trials - Output of buildTrialLists, length t_distinct
 * @param {{design: {trials_per_subject: number, min_trial_repeat_separation: number}}} config
 * @param {function(): number} rng - Seeded RNG (same instance used throughout)
 * @returns {Array<{images: string[], isRepeat: boolean, repeatOfTrialId: (number|null), trialId: (number|string)}>}
 *          Array of length `trials_per_subject`
 * @throws {Error} If no placement satisfies min_trial_repeat_separation
 */
function insertTrialRepeats(trials, config, rng) {
    const t           = config.design.trials_per_subject;
    const numRepeats  = t - trials.length;
    const minSep      = config.design.min_trial_repeat_separation;

    if (numRepeats === 0) {
        return trials.map((images, i) => ({ images, isRepeat: false, repeatOfTrialId: null, trialId: i }));
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

    // Assign originals to repeat slots in increasing position order
    // (most-constrained first), enforcing min_trial_repeat_separation.
    const remaining = seededShuffle([...trials.keys()], rng);
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
