'use strict';

// Requires: utils.js must be loaded before this file (uses seededShuffle).

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Total number of genuinely distinct trial combinations needed across the
 * whole session: sum of each enabled stage's num_experimental_trials
 * (screening_block contributes 0 if disabled). Repeats are additive on top
 * of this, not carved out of it (see insertTrialRepeats).
 * @param {{screening_block: {enabled: boolean, num_experimental_trials: number},
 *          experimental_block: {num_experimental_trials: number}}} config
 * @returns {number}
 */
function _totalDistinctTrialCount(config) {
    const screeningCount = config.screening_block.enabled ? config.screening_block.num_experimental_trials : 0;
    return screeningCount + config.experimental_block.num_experimental_trials;
}

/**
 * Build session-wide trial lists with each image appearing in exactly one
 * distinct trial.
 *
 * The subject sees `_totalDistinctTrialCount(config) * images_per_trial`
 * unique images across that many distinct trials of `images_per_trial`
 * images each. The only way an image can ever appear more than once per
 * subject is via a verbatim whole-trial repeat within a single stage (see
 * insertTrialRepeats), independent of this function. Partitioning this flat
 * list into per-stage groups is handled separately by partitionIntoStages.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{experimental_trials: {images_per_trial: number},
 *          screening_block: {enabled: boolean, num_experimental_trials: number},
 *          experimental_block: {num_experimental_trials: number}}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {string[][]} `_totalDistinctTrialCount(config)` trials of exactly `images_per_trial` images each
 * @throws {Error} If the image pool is too small to fill all slots
 */
function buildTrialLists(allImages, config, rng) {
    const nDistinctTotal = _totalDistinctTrialCount(config);
    const k              = config.experimental_trials.images_per_trial;
    const n_needed        = nDistinctTotal * k;

    if (allImages.length < n_needed) {
        throw new Error(
            `buildTrialLists: image pool has ${allImages.length} image(s), need ${n_needed} ` +
            `(screening_block.num_experimental_trials [if enabled] + experimental_block.num_experimental_trials, ` +
            `× experimental_trials.images_per_trial). Add more images, or lower these counts.`
        );
    }

    // Session-wide random subset, sliced into nDistinctTotal trials of k images each.
    const activeSet = seededShuffle(allImages, rng).slice(0, n_needed);
    const trials    = [];
    for (let i = 0; i < nDistinctTotal; i++) {
        trials.push(activeSet.slice(i * k, (i + 1) * k));
    }

    return trials;
}

/**
 * Partition the session-wide flat list of distinct trials (buildTrialLists'
 * output) into a screening slice and an experimental slice. The screening
 * slice always precedes the experimental slice in the shared pool. Each
 * stage's group is disjoint by construction, since buildTrialLists already
 * guarantees no image is shared between any two distinct trials.
 *
 * @param {string[][]} distinctTrials - length _totalDistinctTrialCount(config)
 * @param {{screening_block: {enabled: boolean, num_experimental_trials: number},
 *          experimental_block: {num_experimental_trials: number}}} config
 * @returns {{screening: string[][], experimental: string[][]}}
 *          screening has length screening_block.num_experimental_trials (0 if disabled);
 *          experimental has length experimental_block.num_experimental_trials
 */
function partitionIntoStages(distinctTrials, config) {
    const screeningCount = config.screening_block.enabled ? config.screening_block.num_experimental_trials : 0;
    return {
        screening:    distinctTrials.slice(0, screeningCount),
        experimental: distinctTrials.slice(screeningCount, screeningCount + config.experimental_block.num_experimental_trials),
    };
}

// ---------------------------------------------------------------------------
// Trial repeats
// ---------------------------------------------------------------------------

/**
 * Fill in `numRepeats` additional slots with verbatim repeats of trials from
 * this same stage, for test-retest reliability of the arrangement response
 * itself.
 *
 * Every distinct trial in the stage is eligible to be repeated (each image
 * already appears in exactly one distinct trial, by construction of
 * buildTrialLists). Each repeat is placed at least `minSep` slots after the
 * trial it duplicates, scoped to this stage's own local slot numbering
 * (image order is reshuffled so the repeat isn't pixel-identical to the
 * original, though the image set is).
 *
 * @param {string[][]} trials - This stage's distinct trials (one stage's slice of buildTrialLists' output)
 * @param {number} numRepeats - Number of additive repeat slots to fill (0..trials.length)
 * @param {number} minSep - Minimum stage-local slot separation between an original and its repeat
 * @param {string} stageLabel - 'screening' or 'experimental', used to namespace trialId across stages
 * @param {function(): number} rng - Seeded RNG (same instance used throughout the session)
 * @returns {Array<{images: string[], isRepeat: boolean, repeatOfTrialId: (string|null), trialId: string}>}
 *          Array of length `trials.length + numRepeats`
 * @throws {Error} If no placement satisfies minSep within this stage
 */
function insertTrialRepeats(trials, numRepeats, minSep, stageLabel, rng) {
    const t = trials.length + numRepeats; // stage-local slot count

    if (numRepeats === 0) {
        return trials.map((images, i) => ({
            images, isRepeat: false, repeatOfTrialId: null, trialId: stageLabel + '_' + i,
        }));
    }

    // Structural layout (which of the t slots are repeats vs. distinct
    // trials) is shared with verifyConfig's static feasibility pre-check —
    // see computeRepeatLayout in utils.js. Stage-scoped: t/numRepeats are
    // this stage's local counts, independent of the other stage.
    const { repeatPositions, distinctPositions } = computeRepeatLayout(t, numRepeats);

    const sequence         = new Array(t).fill(null);
    const originalPosition = {};
    distinctPositions.forEach((pos, distinctIdx) => {
        const trialId = stageLabel + '_' + distinctIdx;
        sequence[pos] = { images: trials[distinctIdx], isRepeat: false, repeatOfTrialId: null, trialId };
        originalPosition[distinctIdx] = pos;
    });

    // Assign originals to repeat slots in increasing position order
    // (most-constrained first), enforcing minSep.
    // Keep the pool keyed by bare distinct-index internally (matching
    // originalPosition's keys) and only namespace into a trialId string
    // when constructing the final repeat object below.
    const remaining = seededShuffle([...trials.keys()], rng);
    for (const pos of repeatPositions) {
        const eligibleIdx = remaining.findIndex(distinctIdx => pos - originalPosition[distinctIdx] >= minSep);
        if (eligibleIdx === -1) {
            throw new Error(
                `insertTrialRepeats: cannot place a repeat at ${stageLabel}-local position ${pos} satisfying ` +
                `min_repeat_separation=${minSep} within ${stageLabel}_block. Reduce num_repeat_trials, ` +
                'lower min_repeat_separation, or increase num_experimental_trials.'
            );
        }
        const distinctIdx = remaining[eligibleIdx];
        remaining.splice(eligibleIdx, 1);
        sequence[pos] = {
            images:          seededShuffle(trials[distinctIdx], rng),
            isRepeat:        true,
            repeatOfTrialId: stageLabel + '_' + distinctIdx,
            trialId:         stageLabel + '_' + distinctIdx + '_repeat',
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
 * Samples `imagesPerTrial` images from the catch pool and draws a target
 * location from CATCH_LOCATIONS — both via the shared seeded RNG so the
 * assignment is reproducible per participant.
 *
 * @param {string[]}          catchPool - Pool of images reserved for catch trials
 * @param {number} imagesPerTrial - Number of images to sample per catch trial
 * @param {function(): number} rng      - Seeded RNG (shared with buildTrialLists)
 * @returns {{type: 'catch', images: string[], target_location: string}}
 */
function buildCatchTrial(catchPool, imagesPerTrial, rng) {
    const images   = seededShuffle(catchPool, rng).slice(0, imagesPerTrial);
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
 * @param {string[]} catchPool  - Images reserved for catch trials
 * @param {number} numCatch - Number of catch trials to interleave
 * @param {number} imagesPerTrial - Images per catch trial
 * @param {function(): number} rng - Seeded RNG (same instance used throughout)
 * @returns {Array<{type: 'main'|'catch', images: string[], target_location?: string, isRepeat?: boolean, repeatOfTrialId?: (number|string|null), trialId?: (number|string)}>}
 */
function insertCatchTrials(mainTrials, catchPool, numCatch, imagesPerTrial, rng) {
    const numMain  = mainTrials.length;

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
            combined.push(buildCatchTrial(catchPool, imagesPerTrial, rng));
            catchIdx++;
        } else {
            combined.push(mainObjects[mainIdx++]);
        }
    }

    return combined;
}

// ---------------------------------------------------------------------------
// Stage orchestration
// ---------------------------------------------------------------------------

/**
 * Compose the stage-scoped repeat + catch logic for a single stage
 * (screening or experimental), and stamp every returned trial (main and
 * catch) with its stage label for downstream data-column output.
 *
 * @param {string[][]} stageDistinctTrials - This stage's slice of num_experimental_trials distinct trials
 *        (one group from partitionIntoStages' output)
 * @param {string[]} catchPool - Images reserved for catch trials
 * @param {{num_repeat_trials: number, min_repeat_separation: number, num_catch_trials: number}} stageConfig
 *        config.screening_block or config.experimental_block
 * @param {{images_per_trial: number}} catchTrialsConfig - config.catch_trials
 * @param {function(): number} rng - Seeded RNG (same instance used throughout the session)
 * @param {string} stageLabel - 'screening' or 'experimental'
 * @returns {Array<{type: 'main'|'catch', images: string[], block: string, [otherFields]: *}>}
 *          Length num_experimental_trials + num_repeat_trials + num_catch_trials
 */
function buildStage(stageDistinctTrials, catchPool, stageConfig, catchTrialsConfig, rng, stageLabel) {
    const mainTrials = insertTrialRepeats(
        stageDistinctTrials, stageConfig.num_repeat_trials, stageConfig.min_repeat_separation, stageLabel, rng,
    );
    const combined = insertCatchTrials(
        mainTrials, catchPool, stageConfig.num_catch_trials, catchTrialsConfig.images_per_trial, rng,
    );
    return combined.map(t => ({ ...t, block: stageLabel }));
}
