'use strict';

/**
 * Compute the structural positions insertTrialRepeats (trial_generator.js)
 * will use for a sequence of `t` main-trial slots with `numRepeats` of them
 * being verbatim repeats — derived from `t` and `numRepeats` alone, with no
 * RNG or trial data involved. Shared so verifyConfig can statically check
 * min_trial_repeat_separation feasibility before any trial is built.
 * @param {number} t
 * @param {number} numRepeats
 * @returns {{ repeatPositions: number[], distinctPositions: number[] }}
 */
function computeRepeatLayout(t, numRepeats) {
    const repeatPositions = [];
    for (let i = 1; i <= numRepeats; i++) {
        let pos = Math.round(t / (numRepeats + 1) * i);
        const prev = repeatPositions[repeatPositions.length - 1];
        if (prev !== undefined && pos <= prev) pos = prev + 1;
        repeatPositions.push(Math.min(pos, t - 1));
    }
    const repeatSet = new Set(repeatPositions);
    const distinctPositions = [];
    for (let pos = 0; pos < t; pos++) {
        if (!repeatSet.has(pos)) distinctPositions.push(pos);
    }
    return { repeatPositions, distinctPositions };
}

/**
 * Best-case feasibility check for min_trial_repeat_separation: true if some
 * assignment of distinct trials to repeat slots could satisfy the separation
 * constraint, assuming every distinct trial were eligible (i.e. singles-only).
 * Necessary but not sufficient — actual runtime feasibility also depends on
 * how many trials end up singles-only, which is data/RNG-dependent and is
 * left to insertTrialRepeats' own runtime check.
 * @param {number} t
 * @param {number} numRepeats
 * @param {number} minSep
 * @returns {boolean}
 */
function canSatisfyTrialRepeatSeparation(t, numRepeats, minSep) {
    if (numRepeats === 0) return true;
    const { repeatPositions, distinctPositions } = computeRepeatLayout(t, numRepeats);
    const available = [...distinctPositions];
    for (const pos of repeatPositions) {
        const idx = available.findIndex(p => pos - p >= minSep);
        if (idx === -1) return false;
        available.splice(idx, 1);
    }
    return true;
}

/**
 * Validate experiment configuration loaded from task_config.json.
 *
 * Throws a descriptive Error on any hard failure (missing/wrong-type key,
 * out-of-range value, or violated arithmetic constraint). Issues console.warn
 * for soft warnings (non-blocking deployment reminders).
 *
 * Call this immediately after parsing task_config.json, before any computation.
 *
 * @param {object} config - Parsed task_config.json object
 * @throws {Error} "verifyConfig: <message>" on hard failure
 */
function verifyConfig(config) {
    const err  = msg => { throw new Error('verifyConfig: ' + msg); };
    const warn = msg => console.warn('verifyConfig: ' + msg);

    // ── Group 1: section presence & key types ────────────────────────────────
    const SCHEMA = {
        design: {
            trials_per_subject:          'number',
            images_per_trial:            'number',
            frac_trials_repeated:        'number',
            min_trial_repeat_separation: 'number',
            min_trial_duration_ms:       'number',
            min_pairwise_distance_sd:    'number',
            min_move_item_ratio:         'number',
            stimuli_path:                'string',
        },
        catch_trials: {
            num_trials:         'number',
            images_per_trial:   'number',
            location_tolerance: 'number',
            stimuli_path:       'string',
        },
        display: {
            sort_area_width:      'number',
            sort_area_height:     'number',
            sort_area_min_width:  'number',
            sort_area_min_height: 'number',
            sort_area_shape:      'string',
            stim_starts_inside:   'boolean',
            column_spread_factor: 'number',
            image_size_fraction:  'number',
            background_color:     'string',
            text_color:           'string',
            font_family:          'string',
            font_size:            'string',
            line_height:          'number',
        },
        deployment: {
            mode:                    'string',
            debug_shine_variant:     'string',
            prolific_completion_url: 'string',
            prolific_no_consent_url: 'string',
        },
        consent: {
            researcher_name:        'string',
            researcher_email:       'string',
            pi_name:                'string',
            pi_email:               'string',
            lab_name:               'string',
            lab_phone:              'string',
            institution:            'string',
            study_duration_minutes: 'number',
        },
    };

    for (const [section, keys] of Object.entries(SCHEMA)) {
        if (!(section in config) || typeof config[section] !== 'object' || config[section] === null)
            err('missing required section "' + section + '".');
        for (const [key, type] of Object.entries(keys)) {
            const fullKey = section + '.' + key;
            if (!(key in config[section]))
                err('missing required key "' + fullKey + '".');
            if (typeof config[section][key] !== type)
                err('"' + fullKey + '" must be a ' + type + ', got ' + typeof config[section][key] + '.');
            if (type === 'number' && !Number.isFinite(config[section][key]))
                err('"' + fullKey + '" must be finite, got ' + config[section][key] + '.');
        }
    }

    // ── Group 1b: consent string fields must not be empty ───────────────────
    const CONSENT_STRING_KEYS = [
        'researcher_name', 'researcher_email',
        'pi_name', 'pi_email',
        'lab_name', 'lab_phone', 'institution',
    ];
    for (const key of CONSENT_STRING_KEYS) {
        if (!config.consent[key].trim())
            err('"consent.' + key + '" must not be empty.');
    }
    if (config.consent.study_duration_minutes <= 0)
        err('"consent.study_duration_minutes" must be > 0, got ' + config.consent.study_duration_minutes + '.');

    // ── Group 2: individual field ranges ─────────────────────────────────────
    const { design: d, catch_trials: ct, display: disp } = config;

    const t          = d.trials_per_subject;
    const k          = d.images_per_trial;
    const fr         = d.frac_trials_repeated;
    const minRepSep  = d.min_trial_repeat_separation;
    const minDur     = d.min_trial_duration_ms;
    const minSd      = d.min_pairwise_distance_sd;
    const moveRatio  = d.min_move_item_ratio;
    const nCatch     = ct.num_trials;
    const kCatch     = ct.images_per_trial;
    const catchTol   = ct.location_tolerance;
    const maxW       = disp.sort_area_width;
    const maxH       = disp.sort_area_height;
    const minW       = disp.sort_area_min_width;
    const minH       = disp.sort_area_min_height;
    const shape      = disp.sort_area_shape;
    const spread     = disp.column_spread_factor;
    const frac       = disp.image_size_fraction;

    if (!Number.isInteger(t)        || t < 1)        err('"design.trials_per_subject" must be a positive integer, got ' + t + '.');
    if (!Number.isInteger(k)        || k < 1)        err('"design.images_per_trial" must be a positive integer, got ' + k + '.');
    if (fr < 0 || fr >= 1)                            err('"design.frac_trials_repeated" must be in [0, 1), got ' + fr + '.');
    if (fr >= 0.4) warn('WARNING: "design.frac_trials_repeated" is ' + fr + ' (>= 0.4). High values leave little room to satisfy "design.min_trial_repeat_separation", raising the risk that insertTrialRepeats fails at runtime. Keep below 0.4 for reliable behaviour.');
    if (!Number.isInteger(minRepSep) || minRepSep < 1) err('"design.min_trial_repeat_separation" must be a positive integer, got ' + minRepSep + '.');
    if (minDur < 0) err('"design.min_trial_duration_ms" must be >= 0, got ' + minDur + '.');
    if (minSd <= 0 || minSd >= 1)     err('"design.min_pairwise_distance_sd" must be in (0, 1), got ' + minSd      + '.');
    if (moveRatio <= 0 || moveRatio > 1) err('"design.min_move_item_ratio" must be in (0, 1], got '   + moveRatio  + '.');
    if (!Number.isInteger(nCatch)   || nCatch < 0)   err('"catch_trials.num_trials" must be a non-negative integer, got ' + nCatch + '.');
    if (!Number.isInteger(kCatch)   || kCatch < 1)   err('"catch_trials.images_per_trial" must be a positive integer, got ' + kCatch + '.');
    if (catchTol  <= 0 || catchTol  >= 1) err('"catch_trials.location_tolerance" must be in (0, 1), got '+ catchTol  + '.');
    if (maxW <= 0) err('"display.sort_area_width" must be > 0, got '        + maxW + '.');
    if (maxH <= 0) err('"display.sort_area_height" must be > 0, got '       + maxH + '.');
    if (minW <= 0) err('"display.sort_area_min_width" must be > 0, got '    + minW + '.');
    if (minH <= 0) err('"display.sort_area_min_height" must be > 0, got '   + minH + '.');
    if (frac  <= 0 || frac  >= 1) err('"display.image_size_fraction" must be in (0, 1), got ' + frac  + '.');
    if (spread <= 0)               err('"display.column_spread_factor" must be > 0, got '      + spread + '.');
    if (shape !== 'rect' && shape !== 'ellipse')
        err('"display.sort_area_shape" must be "rect" or "ellipse", got "' + shape + '".');
    if (!disp.background_color.trim()) err('"display.background_color" must not be empty.');
    if (!disp.text_color.trim())       err('"display.text_color" must not be empty.');
    if (!disp.font_family.trim())      err('"display.font_family" must not be empty.');
    if (!disp.font_size.trim())        err('"display.font_size" must not be empty.');
    if (disp.line_height <= 0)         err('"display.line_height" must be > 0, got ' + disp.line_height + '.');

    // 2b. Deployment mode + debug variant
    const mode = config.deployment.mode;
    if (mode !== 'debug' && mode !== 'pilot' && mode !== 'production')
        err('"deployment.mode" must be "debug", "pilot", or "production", got "' + mode + '".');
    if (mode === 'debug') {
        const dsv = config.deployment.debug_shine_variant;
        if (dsv !== 'pre' && dsv !== 'post')
            err('"deployment.debug_shine_variant" must be "pre" or "post", got "' + dsv + '".');
    }

    // ── Group 3: cross-parameter arithmetic ──────────────────────────────────

    // 3a. min_trial_repeat_separation structural feasibility. The slot
    // positions (which of the t main-trial slots are repeats) are fully
    // determined by t and numTrialRepeats alone — independent of the RNG —
    // so this can be checked at config-load time. Failing here means
    // insertTrialRepeats is guaranteed to fail too.
    const tDistinct       = t - Math.round(fr * t);
    const numTrialRepeats = t - tDistinct;
    if (!canSatisfyTrialRepeatSeparation(t, numTrialRepeats, minRepSep))
        err('design.min_trial_repeat_separation (' + minRepSep + ') cannot be satisfied with ' +
            'trials_per_subject=' + t + ' and frac_trials_repeated=' + fr + ' (' + numTrialRepeats +
            ' repeat slot(s) needed). Lower min_trial_repeat_separation, lower frac_trials_repeated, ' +
            'or increase trials_per_subject. Note: catch trials are not counted toward this separation.');

    // 3b. Catch trial count
    if (nCatch >= t)
        err('catch_trials.num_trials (' + nCatch + ') must be < design.trials_per_subject (' + t + '). At least one main trial is required.');

    // 3c. Sort area min <= max
    if (minW > maxW) err('display.sort_area_min_width ('  + minW + ') must not exceed display.sort_area_width ('  + maxW + ').');
    if (minH > maxH) err('display.sort_area_min_height (' + minH + ') must not exceed display.sort_area_height (' + maxH + ').');

    // 3d. Single image fits in minimum sort area
    const stimSize = Math.round(minW * frac);
    if (stimSize >= minW) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds display.sort_area_min_width ('  + minW + 'px). Reduce display.image_size_fraction.');
    if (stimSize >= minH) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds display.sort_area_min_height (' + minH + 'px). Reduce display.image_size_fraction.');

    // 3e. k images fit in a square grid within the minimum sort area
    const colsMain = Math.ceil(Math.sqrt(k));
    const rowsMain = Math.ceil(k / colsMain);
    if (colsMain * stimSize > minW)
        err(k + ' images of ' + stimSize + 'px each cannot fit in a ' + colsMain + '×' + rowsMain +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce display.image_size_fraction or design.images_per_trial, or increase sort area dimensions.');
    if (rowsMain * stimSize > minH)
        err(k + ' images of ' + stimSize + 'px each cannot fit in a ' + colsMain + '×' + rowsMain +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce display.image_size_fraction or design.images_per_trial, or increase sort area dimensions.');

    // 3e (catch). kCatch images fit in a square grid within the minimum sort area
    const colsCatch = Math.ceil(Math.sqrt(kCatch));
    const rowsCatch = Math.ceil(kCatch / colsCatch);
    if (colsCatch * stimSize > minW)
        err(kCatch + ' catch images of ' + stimSize + 'px each cannot fit in a ' + colsCatch + '×' + rowsCatch +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce display.image_size_fraction or catch_trials.images_per_trial, or increase sort area dimensions.');
    if (rowsCatch * stimSize > minH)
        err(kCatch + ' catch images of ' + stimSize + 'px each cannot fit in a ' + colsCatch + '×' + rowsCatch +
            ' grid within the minimum sort area (' + minW + '×' + minH + 'px). Reduce display.image_size_fraction or catch_trials.images_per_trial, or increase sort area dimensions.');

    // ── Group 4: deployment warnings ─────────────────────────────────────────
    if (mode !== 'debug') {
        if (!config.design.stimuli_path)                   warn('"design.stimuli_path" is empty — no main images will load.');
        if (!config.catch_trials.stimuli_path)             warn('"catch_trials.stimuli_path" is empty — catch and practice trials will have no images.');
        if (!config.deployment.prolific_completion_url)    warn('"deployment.prolific_completion_url" is empty — participants will not be redirected after completion.');
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
 * @param {{ display: {
 *   sort_area_width: number, sort_area_height: number,
 *   sort_area_min_width: number, sort_area_min_height: number,
 *   image_size_fraction: number,
 *   stim_starts_inside: boolean, column_spread_factor: number
 * }}} config
 * @returns {{ sortW: number, sortH: number, stimSize: number }}
 */
function computeLayout(viewportW, viewportH, config) {
    const {
        sort_area_width, sort_area_height,
        sort_area_min_width, sort_area_min_height,
        image_size_fraction, stim_starts_inside, column_spread_factor,
    } = config.display;

    const maxSortW = stim_starts_inside
        ? Math.floor(viewportW * 0.85)
        : Math.floor(viewportW / (1 + column_spread_factor));
    const sortW = Math.max(
        sort_area_min_width,
        Math.min(maxSortW, sort_area_width),
    );
    const sortH = Math.max(
        sort_area_min_height,
        Math.min(Math.floor(viewportH * 0.70), sort_area_height),
    );
    const stimSize = Math.round(sortW * image_size_fraction);
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
    const EDGE = 0.05; // fraction from edge for corner targets
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
 * Check whether EVERY image in a catch trial is within tolerance of the target
 * location. Used for both the real-time blocking check (on_load) and the
 * post-trial QC flag (on_finish), so both use identical criteria.
 *
 * Proximity for each image is the Euclidean distance from the image position
 * to the target point, normalised by the sort-area diagonal — same
 * normalisation as pairwise distances — so tolerance is resolution-independent.
 *
 * @param {Array<{src: string, x: number, y: number}>} locations
 * @param {string}  targetLocation - One of the CATCH_LOCATIONS strings
 * @param {number}  sortAreaWidth
 * @param {number}  sortAreaHeight
 * @param {number}  tolerance - Max normalised per-image distance (e.g. 0.25)
 * @returns {boolean} true iff every image is within tolerance of the target point
 */
function allImagesNearTarget(locations, targetLocation, sortAreaWidth, sortAreaHeight, tolerance) {
    if (locations.length === 0) return false;
    const target   = _targetPoint(targetLocation, sortAreaWidth, sortAreaHeight);
    const diagonal = Math.sqrt(sortAreaWidth * sortAreaWidth + sortAreaHeight * sortAreaHeight);
    return locations.every(loc => {
        const dist = Math.sqrt(
            (loc.x - target.x) ** 2 +
            (loc.y - target.y) ** 2
        ) / diagonal;
        return dist <= tolerance;
    });
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

/**
 * QC flag for a main (non-catch) trial.
 * Flags if the participant barely moved items OR piled everything together.
 *
 * @param {number} sd          - Sample SD of normalised pairwise distances
 * @param {number} numMoves    - Total drag-end events recorded by the plugin
 * @param {number} numItems    - Number of stimuli in the trial
 * @param {object} config      - Task config (reads design.min_pairwise_distance_sd / design.min_move_item_ratio)
 * @returns {boolean} true if the trial should be flagged
 */
function computeMainQcFlag(sd, numMoves, numItems, config) {
    const d = config.design;
    const enoughMoves = numMoves >= d.min_move_item_ratio * numItems;
    return sd < d.min_pairwise_distance_sd || !enoughMoves;
}
