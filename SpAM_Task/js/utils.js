'use strict';

/**
 * djb2 hash: converts a string to a positive 32-bit integer.
 * Used to derive a numeric seed from a Prolific PID so every subject
 * gets a deterministic but unique stimulus assignment.
 *
 * @param {string} str - Input string (e.g. Prolific PID)
 * @returns {number} Positive integer hash
 */
function hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) + str.charCodeAt(i);
        hash |= 0; // keep within 32-bit integer range
    }
    return Math.abs(hash);
}

/**
 * Fisher-Yates shuffle using a caller-supplied seeded RNG.
 * Does NOT mutate the input array.
 *
 * @param {Array} array - Array to shuffle
 * @param {function(): number} rng - Seeded RNG returning a float in [0, 1)
 * @returns {Array} New shuffled array
 */
function seededShuffle(array, rng) {
    const result = array.slice();
    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = result[i];
        result[i] = result[j];
        result[j] = tmp;
    }
    return result;
}

/**
 * Compute normalised pairwise Euclidean distances for all image locations
 * from a jsPsych free-sort trial.
 *
 * Distances are divided by the sort-area diagonal √(w²+h²) so that values
 * fall in [0, 1] regardless of screen resolution or sort-area size.
 *
 * @param {Array<{src: string, x: number, y: number}>} locations
 *   jsPsych free-sort `final_locations` output
 * @param {number} sortAreaWidth  - Sort canvas width in pixels
 * @param {number} sortAreaHeight - Sort canvas height in pixels
 * @returns {Array<{src1: string, src2: string, distance: number}>}
 *   One entry per unique pair (i < j), distance normalised to [0, 1]
 */
function computePairwiseDistances(locations, sortAreaWidth, sortAreaHeight) {
    const diagonal = Math.sqrt(
        sortAreaWidth  * sortAreaWidth +
        sortAreaHeight * sortAreaHeight
    );
    const pairs = [];
    for (let i = 0; i < locations.length; i++) {
        for (let j = i + 1; j < locations.length; j++) {
            const dx = locations[i].x - locations[j].x;
            const dy = locations[i].y - locations[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy) / diagonal;
            pairs.push({ src1: locations[i].src, src2: locations[j].src, distance });
        }
    }
    return pairs;
}

/**
 * Compute the centroid (mean position) of a set of placed images.
 *
 * @param {Array<{src: string, x: number, y: number}>} locations
 *   jsPsych free-sort `final_locations` output
 * @returns {{x: number, y: number}} Mean x and y coordinates
 */
function computeCentroid(locations) {
    if (locations.length === 0) return { x: 0, y: 0 };
    const sumX = locations.reduce((acc, loc) => acc + loc.x, 0);
    const sumY = locations.reduce((acc, loc) => acc + loc.y, 0);
    return { x: sumX / locations.length, y: sumY / locations.length };
}

/**
 * Target zone definitions for catch trials.
 * Returns the expected centroid (x, y) as a fraction of sort-area dimensions.
 * "corner" zones are set at 15% from each edge; centre is 50/50.
 *
 * @param {string} targetLocation - One of the CATCH_LOCATIONS strings
 * @param {number} sortAreaWidth
 * @param {number} sortAreaHeight
 * @returns {{x: number, y: number}} Absolute pixel coordinates of target point
 */
function _targetPoint(targetLocation, sortAreaWidth, sortAreaHeight) {
    const EDGE = 0.15; // fraction from edge for corner targets
    const map  = {
        'center':             { fx: 0.50, fy: 0.50 },
        'top left corner':    { fx: EDGE,      fy: EDGE       },
        'top right corner':   { fx: 1 - EDGE,  fy: EDGE       },
        'bottom left corner': { fx: EDGE,      fy: 1 - EDGE   },
        'bottom right corner':{ fx: 1 - EDGE,  fy: 1 - EDGE   },
    };
    const { fx, fy } = map[targetLocation] || map['center'];
    return { x: fx * sortAreaWidth, y: fy * sortAreaHeight };
}

/**
 * Check whether an image cluster's centroid is close enough to the target location.
 *
 * Proximity is measured as the Euclidean distance between the centroid and the
 * target point, normalised by the sort-area diagonal — the same normalisation
 * used for pairwise distances — so the tolerance is resolution-independent.
 *
 * @param {{x: number, y: number}} centroid  - Output of computeCentroid
 * @param {string}  targetLocation           - One of the CATCH_LOCATIONS strings
 * @param {number}  sortAreaWidth
 * @param {number}  sortAreaHeight
 * @param {number}  tolerance               - Max normalised distance (e.g. 0.20)
 * @returns {boolean}
 */
function isCentroidNearTarget(centroid, targetLocation, sortAreaWidth, sortAreaHeight, tolerance) {
    const target   = _targetPoint(targetLocation, sortAreaWidth, sortAreaHeight);
    const diagonal = Math.sqrt(sortAreaWidth * sortAreaWidth + sortAreaHeight * sortAreaHeight);
    const dist     = Math.sqrt(
        (centroid.x - target.x) ** 2 +
        (centroid.y - target.y) ** 2
    ) / diagonal;
    return dist <= tolerance;
}

/**
 * Sample standard deviation (denominator n−1).
 * Used for QC: flag a trial if SD of normalised distances is below threshold,
 * which indicates the participant placed all images in a cluster.
 *
 * @param {number[]} values - Array of numbers
 * @returns {number} Sample SD, or 0 if fewer than 2 values
 */
function computeSD(values) {
    if (values.length < 2) return 0;
    const n    = values.length;
    const mean = values.reduce((acc, v) => acc + v, 0) / n;
    const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1);
    return Math.sqrt(variance);
}
