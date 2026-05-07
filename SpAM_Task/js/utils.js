'use strict';

/**
 * Validate experiment configuration loaded from config.json.
 *
 * Throws a descriptive Error on any hard failure (missing/wrong-type key,
 * out-of-range value, or violated arithmetic constraint). Issues console.warn
 * for soft warnings (non-blocking deployment reminders).
 *
 * Call this immediately after parsing config.json, before any computation.
 *
 * @param {object} config - Parsed config.json object
 * @throws {Error} "verifyConfig: <message>" on hard failure
 */
function verifyConfig(config) {
    const err  = msg => { throw new Error('verifyConfig: ' + msg); };
    const warn = msg => console.warn('verifyConfig: ' + msg);

    // ── Group 1: key presence & types ────────────────────────────────────────
    const REQUIRED_TYPES = {
        trials_per_subject:        'number',
        images_per_trial:          'number',
        unique_images_per_subject: 'number',
        num_catch_trials:          'number',
        catch_images_per_trial:    'number',
        practice_images_per_trial: 'number',
        sort_area_width:           'number',
        sort_area_height:          'number',
        sort_area_min_width:       'number',
        sort_area_min_height:      'number',
        image_size_fraction:       'number',
        min_trial_rt_ms:           'number',
        min_pairwise_distance_sd:  'number',
        catch_cluster_max_mean:    'number',
        catch_cluster_max_sd:      'number',
        catch_location_tolerance:  'number',
        sort_area_shape:           'string',
        stim_starts_inside:        'boolean',
        column_spread_factor:      'number',
        debug:                     'boolean',
    };
    for (const [key, type] of Object.entries(REQUIRED_TYPES)) {
        if (!(key in config))
            err('missing required key "' + key + '".');
        if (typeof config[key] !== type)
            err('"' + key + '" must be a ' + type + ', got ' + typeof config[key] + '.');
        if (type === 'number' && !Number.isFinite(config[key]))
            err('"' + key + '" must be finite, got ' + config[key] + '.');
    }

    // ── Group 2: individual field ranges ─────────────────────────────────────
    const {
        trials_per_subject: t,
        images_per_trial: k,
        unique_images_per_subject: N,
        num_catch_trials: nCatch,
        catch_images_per_trial: kCatch,
        practice_images_per_trial: kPractice,
        sort_area_width: maxW,    sort_area_height: maxH,
        sort_area_min_width: minW, sort_area_min_height: minH,
        image_size_fraction: frac,
        min_trial_rt_ms: minRt,
        min_pairwise_distance_sd: minSd,
        catch_cluster_max_mean: catchMean,
        catch_cluster_max_sd:   catchSd,
        catch_location_tolerance: catchTol,
        sort_area_shape: shape,
        column_spread_factor: spreadFactor,
    } = config;

    if (!Number.isInteger(t)        || t < 1)        err('"trials_per_subject" must be a positive integer, got ' + t + '.');
    if (!Number.isInteger(k)        || k < 1)        err('"images_per_trial" must be a positive integer, got ' + k + '.');
    if (!Number.isInteger(N)        || N < 1)        err('"unique_images_per_subject" must be a positive integer, got ' + N + '.');
    if (!Number.isInteger(nCatch)   || nCatch < 0)   err('"num_catch_trials" must be a non-negative integer, got ' + nCatch + '.');
    if (!Number.isInteger(kCatch)   || kCatch < 1)   err('"catch_images_per_trial" must be a positive integer, got ' + kCatch + '.');
    if (!Number.isInteger(kPractice)|| kPractice < 1)err('"practice_images_per_trial" must be a positive integer, got ' + kPractice + '.');
    if (maxW   <= 0) err('"sort_area_width" must be > 0, got '        + maxW + '.');
    if (maxH   <= 0) err('"sort_area_height" must be > 0, got '       + maxH + '.');
    if (minW   <= 0) err('"sort_area_min_width" must be > 0, got '    + minW + '.');
    if (minH   <= 0) err('"sort_area_min_height" must be > 0, got '   + minH + '.');
    if (frac <= 0 || frac >= 1)      err('"image_size_fraction" must be in (0, 1), got '            + frac + '.');
    if (minRt < 0)                   err('"min_trial_rt_ms" must be >= 0, got '                     + minRt + '.');
    if (minSd <= 0 || minSd >= 1)    err('"min_pairwise_distance_sd" must be in (0, 1), got '       + minSd + '.');
    if (catchMean <= 0 || catchMean >= 1) err('"catch_cluster_max_mean" must be in (0, 1), got '    + catchMean + '.');
    if (catchSd   <= 0 || catchSd   >= 1) err('"catch_cluster_max_sd" must be in (0, 1), got '      + catchSd + '.');
    if (catchTol  <= 0 || catchTol  >= 1) err('"catch_location_tolerance" must be in (0, 1), got '  + catchTol + '.');
    if (shape !== 'rect' && shape !== 'ellipse')
        err('"sort_area_shape" must be "rect" or "ellipse", got "' + shape + '".');
    if (spreadFactor <= 0)
        err('"column_spread_factor" must be > 0, got ' + spreadFactor + '.');

    // ── Group 3: cross-parameter arithmetic ──────────────────────────────────

    // 3a. Trial image pool arithmetic
    if (N < k)
        err('unique_images_per_subject (' + N + ') must be >= images_per_trial (' + k + '). Each trial needs k distinct images.');
    if (N > t * k)
        err('unique_images_per_subject (' + N + ') exceeds trials_per_subject × images_per_trial (t×k = ' + (t * k) + '). Reduce N or increase t or k.');

    // 3b. Catch trial count
    if (nCatch >= t)
        err('num_catch_trials (' + nCatch + ') must be < trials_per_subject (' + t + '). At least one main trial is required.');

    // 3c. Sort area min <= max
    if (minW > maxW) err('sort_area_min_width (' + minW + ') must not exceed sort_area_width (' + maxW + ').');
    if (minH > maxH) err('sort_area_min_height (' + minH + ') must not exceed sort_area_height (' + maxH + ').');

    // 3d. Single image fits in minimum sort area
    const stimSize = Math.round(minW * frac);
    if (stimSize >= minW) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds sort_area_min_width (' + minW + 'px). Reduce image_size_fraction.');
    if (stimSize >= minH) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds sort_area_min_height (' + minH + 'px). Reduce image_size_fraction.');

    // 3e. k images fit in a square grid within the minimum sort area
    const colsMain  = Math.ceil(Math.sqrt(k));
    const rowsMain  = Math.ceil(k / colsMain);
    if (colsMain * stimSize > minW)
        err(k + ' images of ' + stimSize + 'px each cannot fit in a ' + colsMain + '×' + rowsMain +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce image_size_fraction or images_per_trial, or increase sort area dimensions.');
    if (rowsMain * stimSize > minH)
        err(k + ' images of ' + stimSize + 'px each cannot fit in a ' + colsMain + '×' + rowsMain +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce image_size_fraction or images_per_trial, or increase sort area dimensions.');

    // 3e (catch). kCatch images fit in a square grid within the minimum sort area
    const colsCatch = Math.ceil(Math.sqrt(kCatch));
    const rowsCatch = Math.ceil(kCatch / colsCatch);
    if (colsCatch * stimSize > minW)
        err(kCatch + ' catch images of ' + stimSize + 'px each cannot fit in a ' + colsCatch + '×' + rowsCatch +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce image_size_fraction or catch_images_per_trial, or increase sort area dimensions.');
    if (rowsCatch * stimSize > minH)
        err(kCatch + ' catch images of ' + stimSize + 'px each cannot fit in a ' + colsCatch + '×' + rowsCatch +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce image_size_fraction or catch_images_per_trial, or increase sort area dimensions.');

    // 3f. Practice image count (soft warning)
    if (kPractice > k)
        warn('"practice_images_per_trial" (' + kPractice + ') > "images_per_trial" (' + k + '). Practice trial will show more images than main trials.');

    // ── Group 4: deployment warnings ─────────────────────────────────────────
    if (!config.debug) {
        if (!config.stimuli_path)           warn('"stimuli_path" is empty — no main images will load.');
        if (!config.stimuli_practice_path)  warn('"stimuli_practice_path" is empty — practice trial will have no images.');
        if (!config.stimuli_catch_path)     warn('"stimuli_catch_path" is empty — catch trials will have no images.');
        if (!config.prolific_completion_url)warn('"prolific_completion_url" is empty — participants will not be redirected after completion.');
    }
}

/**
 * Compute the sort-area dimensions and stimulus size for the current viewport.
 *
 * Sort area is clamped between [min, max] on each axis, where max is the ideal
 * configured size and min is the smallest acceptable size. Image size is expressed
 * as a fraction of the sort-area width so that emoji and dataset images are always
 * rendered at the same size regardless of their native resolution.
 *
 * When `stim_starts_inside` is true, images begin inside the sort area so the canvas
 * can fill 85% of the viewport width. When false, images start in staging columns to
 * the left and right of the sort area; the plugin positions them at
 * ±(sortW × 0.5 × column_spread_factor) from the arena edge, so the total horizontal
 * space needed is sortW × (1 + column_spread_factor). In that case the canvas width is
 * capped at floor(viewportW / (1 + column_spread_factor)) to prevent overflow.
 * Height is clamped to 70% of viewport height to leave room for the prompt, counter,
 * and done button above/below the sort area without triggering a vertical scrollbar.
 *
 * @param {number} viewportW - window.innerWidth
 * @param {number} viewportH - window.innerHeight
 * @param {{
 *   sort_area_width: number, sort_area_height: number,
 *   sort_area_min_width: number, sort_area_min_height: number,
 *   image_size_fraction: number,
 *   stim_starts_inside: boolean, column_spread_factor: number
 * }} config
 * @returns {{ sortW: number, sortH: number, stimSize: number }}
 */
function computeLayout(viewportW, viewportH, config) {
    const maxSortW = config.stim_starts_inside
        ? Math.floor(viewportW * 0.85)
        : Math.floor(viewportW / (1 + config.column_spread_factor));
    const sortW = Math.max(
        config.sort_area_min_width,
        Math.min(maxSortW, config.sort_area_width),
    );
    const sortH = Math.max(
        config.sort_area_min_height,
        Math.min(Math.floor(viewportH * 0.70), config.sort_area_height),
    );
    const stimSize = Math.round(sortW * config.image_size_fraction);
    return { sortW, sortH, stimSize };
}

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
