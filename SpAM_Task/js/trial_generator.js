'use strict';

// Requires: utils.js must be loaded before this file (uses seededShuffle).

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Total number of genuinely distinct trial combinations needed across the
 * whole session: every block contributes trials_per_block distinct trials,
 * and (unlike the old per-session frac_trials_repeated scheme) repeats are
 * additive on top of that, not carved out of it.
 * @param {{design: {num_blocks: number, trials_per_block: number}}} config
 * @returns {number}
 */
function _totalDistinctTrialCount(config) {
    return config.design.num_blocks * config.design.trials_per_block;
}

/**
 * Build session-wide trial lists with each image appearing in exactly one
 * distinct trial.
 *
 * The subject sees (num_blocks * trials_per_block) * images_per_trial unique
 * images across that many distinct trials of images_per_trial images each.
 * The only way an image can ever appear more than once per subject is via a
 * verbatim whole-trial repeat within a single block (see insertTrialRepeats),
 * independent of this function. Partitioning this flat list into per-block
 * groups is handled separately by partitionIntoBlocks.
 *
 * @param {string[]} allImages - All available image paths (from stimuli_manifest.json)
 * @param {{design: {num_blocks: number,
 *                   trials_per_block: number,
 *                   images_per_trial: number}}} config
 * @param {function(): number} rng - Seeded RNG returning float in [0, 1)
 * @returns {string[][]} `num_blocks * trials_per_block` trials of exactly `images_per_trial` images each
 * @throws {Error} If the image pool is too small to fill all slots
 */
function buildTrialLists(allImages, config, rng) {
    const nDistinctTotal = _totalDistinctTrialCount(config);
    const k              = config.design.images_per_trial;
    const n_needed        = nDistinctTotal * k;

    if (allImages.length < n_needed) {
        throw new Error(
            `buildTrialLists: image pool has ${allImages.length} image(s), need ${n_needed} ` +
            `(num_blocks × trials_per_block × images_per_trial). ` +
            'Add more images, or lower num_blocks/trials_per_block/images_per_trial.'
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
 * output) into num_blocks consecutive groups of trials_per_block each. Each
 * block's group is disjoint by construction, since buildTrialLists already
 * guarantees no image is shared between any two distinct trials.
 *
 * @param {string[][]} distinctTrials - length num_blocks * trials_per_block
 * @param {{design: {num_blocks: number, trials_per_block: number}}} config
 * @returns {string[][][]} length num_blocks, each element length trials_per_block
 */
function partitionIntoBlocks(distinctTrials, config) {
    const B = config.design.num_blocks;
    const k = config.design.trials_per_block;
    const blocks = [];
    for (let b = 0; b < B; b++) {
        blocks.push(distinctTrials.slice(b * k, (b + 1) * k));
    }
    return blocks;
}

// ---------------------------------------------------------------------------
// Trial repeats
// ---------------------------------------------------------------------------

/**
 * Fill in `repeats_per_block` additional slots with verbatim repeats of
 * trials from this same block, for test-retest reliability of the
 * arrangement response itself.
 *
 * Every distinct trial in the block is eligible to be repeated (each image
 * already appears in exactly one distinct trial, by construction of
 * buildTrialLists). Each repeat is placed at least `min_trial_repeat_separation`
 * slots after the trial it duplicates, scoped to this block's own local
 * slot numbering (image order is reshuffled so the repeat isn't
 * pixel-identical to the original, though the image set is).
 *
 * @param {string[][]} trials - This block's distinct trials (one block's slice of buildTrialLists' output)
 * @param {{design: {repeats_per_block: number, min_trial_repeat_separation: number}}} config
 * @param {function(): number} rng - Seeded RNG (same instance used throughout the session)
 * @param {number} [blockIndex=0] - 0-based block index, used to namespace trialId globally across blocks
 * @returns {Array<{images: string[], isRepeat: boolean, repeatOfTrialId: (string|null), trialId: string}>}
 *          Array of length `trials.length + repeats_per_block`
 * @throws {Error} If no placement satisfies min_trial_repeat_separation within this block
 */
function insertTrialRepeats(trials, config, rng, blockIndex = 0) {
    const numRepeats = config.design.repeats_per_block;
    const t          = trials.length + numRepeats; // block-local slot count
    const minSep     = config.design.min_trial_repeat_separation;

    if (numRepeats === 0) {
        return trials.map((images, i) => ({
            images, isRepeat: false, repeatOfTrialId: null, trialId: blockIndex + '_' + i,
        }));
    }

    // Structural layout (which of the t slots are repeats vs. distinct
    // trials) is shared with verifyConfig's static feasibility pre-check —
    // see computeRepeatLayout in utils.js. Block-scoped: t/numRepeats are
    // this block's local counts, not session-wide.
    const { repeatPositions, distinctPositions } = computeRepeatLayout(t, numRepeats);

    const sequence         = new Array(t).fill(null);
    const originalPosition = {};
    distinctPositions.forEach((pos, distinctIdx) => {
        const trialId = blockIndex + '_' + distinctIdx;
        sequence[pos] = { images: trials[distinctIdx], isRepeat: false, repeatOfTrialId: null, trialId };
        originalPosition[distinctIdx] = pos;
    });

    // Assign originals to repeat slots in increasing position order
    // (most-constrained first), enforcing min_trial_repeat_separation.
    // Keep the pool keyed by bare distinct-index internally (matching
    // originalPosition's keys) and only namespace into a trialId string
    // when constructing the final repeat object below.
    const remaining = seededShuffle([...trials.keys()], rng);
    for (const pos of repeatPositions) {
        const eligibleIdx = remaining.findIndex(distinctIdx => pos - originalPosition[distinctIdx] >= minSep);
        if (eligibleIdx === -1) {
            throw new Error(
                `insertTrialRepeats: cannot place a repeat at block-local position ${pos} satisfying ` +
                `min_trial_repeat_separation=${minSep} within block ${blockIndex}. Reduce repeats_per_block, ` +
                'lower min_trial_repeat_separation, or increase trials_per_block.'
            );
        }
        const distinctIdx = remaining[eligibleIdx];
        remaining.splice(eligibleIdx, 1);
        sequence[pos] = {
            images:          seededShuffle(trials[distinctIdx], rng),
            isRepeat:        true,
            repeatOfTrialId: blockIndex + '_' + distinctIdx,
            trialId:         blockIndex + '_' + distinctIdx + '_repeat',
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
 * @param {{catch_trials: {catch_per_block: number, images_per_trial: number}}} config
 * @param {function(): number} rng        - Seeded RNG (same instance used throughout)
 * @returns {Array<{type: 'main'|'catch', images: string[], target_location?: string, isRepeat?: boolean, repeatOfTrialId?: (number|string|null), trialId?: (number|string)}>}
 */
function insertCatchTrials(mainTrials, catchPool, config, rng) {
    const numMain  = mainTrials.length;
    const numCatch = config.catch_trials.catch_per_block;

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

// ---------------------------------------------------------------------------
// Block orchestration
// ---------------------------------------------------------------------------

/**
 * Compose the block-scoped repeat + catch logic for a single block, and
 * stamp every returned trial (main and catch) with its 1-based block number
 * for downstream data-column output.
 *
 * @param {string[][]} blockDistinctTrials - This block's slice of trials_per_block distinct trials
 *        (one group from partitionIntoBlocks' output)
 * @param {string[]} catchPool - Images reserved for catch trials
 * @param {object} config - Full task config (design.repeats_per_block, design.min_trial_repeat_separation,
 *        catch_trials.catch_per_block, catch_trials.images_per_trial)
 * @param {function(): number} rng - Seeded RNG (same instance used throughout the session)
 * @param {number} blockIndex - 0-based block index
 * @returns {Array<{type: 'main'|'catch', images: string[], block: number, [otherFields]: *}>}
 *          Length trials_per_block + repeats_per_block + catch_per_block
 */
function buildBlock(blockDistinctTrials, catchPool, config, rng, blockIndex) {
    const mainTrials = insertTrialRepeats(blockDistinctTrials, config, rng, blockIndex);
    const combined    = insertCatchTrials(mainTrials, catchPool, config, rng);
    return combined.map(t => ({ ...t, block: blockIndex + 1 }));
}
