'use strict';

/**
 * Round a float to at most `decimals` decimal places, avoiding the
 * spurious 16-18 digit tails that raw IEEE-754 arithmetic (pixel divisions,
 * correlation coefficients, aggregated rates) otherwise leaves in output
 * data. Non-finite values (NaN, ±Infinity) pass through unchanged, since
 * `reliability` is legitimately NaN when a repeat has no comparable original.
 *
 * @param {number} value
 * @param {number} [decimals=10]
 * @returns {number}
 */
function roundTo(value, decimals = 10) {
    if (!Number.isFinite(value)) return value;
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
}

/**
 * Compute the structural positions insertTrialRepeats (trial_generator.js)
 * will use for a sequence of `t` main-trial slots with `numRepeats` of them
 * being verbatim repeats — derived from `t` and `numRepeats` alone, with no
 * RNG or trial data involved. Shared so verifyConfig can statically check
 * min_repeat_separation feasibility before any trial is built.
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
 * Best-case feasibility check for min_repeat_separation: true if some
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
        experimental_trials: {
            stimuli_path:               'string',
            images_per_trial:           'number',
            min_trial_duration_ms:      'number',
            min_move_item_ratio:        'number',
            min_pairwise_distance_sd:   'number',
        },
        catch_trials: {
            stimuli_path:       'string',
            images_per_trial:   'number',
            location_tolerance: 'number',
        },
        screening_block: {
            enabled:                  'boolean',
            prolific_code:            'string',
            num_catch_trials:         'number',
            num_experimental_trials:  'number',
            num_repeat_trials:        'number',
            min_repeat_separation:    'number',
            // thresholds is a nested object whose 4 fields may each be
            // `null` (disabled) or a number — validated separately below
            // (outside the generic scalar-type-check loop).
        },
        experimental_block: {
            prolific_code:            'string',
            num_catch_trials:         'number',
            num_experimental_trials:  'number',
            num_repeat_trials:        'number',
            min_repeat_separation:    'number',
        },
        display: {
            sort_area_min_width:  'number',
            sort_area_min_height: 'number',
            min_header_height_px: 'number',
            min_footer_height_px: 'number',
            image_size_fraction:  'number',
            background_color:     'string',
            text_color:           'string',
            font_family:          'string',
            font_size:            'string',
            line_height:          'number',
        },
        deployment: {
            mode:                'string',
            debug_shine_variant: 'string',
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
            no_consent_code:        'string',
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

    // ── Group 1c: screening_block.thresholds (nested, number|null fields, validated outside the generic loop) ──
    if (!('thresholds' in config.screening_block) || typeof config.screening_block.thresholds !== 'object' || config.screening_block.thresholds === null)
        err('missing required section "screening_block.thresholds".');
    const THRESHOLD_KEYS = ['min_reliability', 'median_reliability', 'move_ratio_max_fail_rate', 'distance_sd_max_fail_rate'];
    for (const key of THRESHOLD_KEYS) {
        const fullKey = 'screening_block.thresholds.' + key;
        if (!(key in config.screening_block.thresholds))
            err('missing required key "' + fullKey + '".');
        const v = config.screening_block.thresholds[key];
        if (v !== null && typeof v !== 'number')
            err('"' + fullKey + '" must be a number or null, got ' + typeof v + '.');
        if (typeof v === 'number' && !Number.isFinite(v))
            err('"' + fullKey + '" must be finite, got ' + v + '.');
    }

    // ── Group 2: individual field ranges ─────────────────────────────────────
    const { experimental_trials: et, catch_trials: ct, display: disp, screening_block: sb, experimental_block: eb } = config;

    const k               = et.images_per_trial;
    const minDur          = et.min_trial_duration_ms;
    const trialMinSd      = et.min_pairwise_distance_sd;
    const trialMoveRatio  = et.min_move_item_ratio;
    const kCatch          = ct.images_per_trial;
    const catchTol        = ct.location_tolerance;
    const minW            = disp.sort_area_min_width;
    const minH            = disp.sort_area_min_height;
    const minHeader       = disp.min_header_height_px;
    const minFooter       = disp.min_footer_height_px;
    const frac            = disp.image_size_fraction;

    if (!Number.isInteger(k)        || k < 1)        err('"experimental_trials.images_per_trial" must be a positive integer, got ' + k + '.');
    if (minDur < 0) err('"experimental_trials.min_trial_duration_ms" must be >= 0, got ' + minDur + '.');
    if (trialMinSd <= 0 || trialMinSd >= 1)     err('"experimental_trials.min_pairwise_distance_sd" must be in (0, 1), got ' + trialMinSd      + '.');
    if (trialMoveRatio <= 0 || trialMoveRatio > 1) err('"experimental_trials.min_move_item_ratio" must be in (0, 1], got '   + trialMoveRatio  + '.');
    if (!Number.isInteger(kCatch)   || kCatch < 1)   err('"catch_trials.images_per_trial" must be a positive integer, got ' + kCatch + '.');
    if (catchTol  <= 0 || catchTol  >= 1) err('"catch_trials.location_tolerance" must be in (0, 1), got '+ catchTol  + '.');

    // 2a. screening_block / experimental_block, validated identically except
    // screening_block additionally has thresholds (checked separately below).
    const STAGES = [['screening_block', sb], ['experimental_block', eb]];
    for (const [name, stage] of STAGES) {
        if (!Number.isInteger(stage.num_experimental_trials) || stage.num_experimental_trials < 1)
            err('"' + name + '.num_experimental_trials" must be a positive integer, got ' + stage.num_experimental_trials + '.');
        if (!Number.isInteger(stage.num_repeat_trials) || stage.num_repeat_trials < 0 || stage.num_repeat_trials > stage.num_experimental_trials)
            err('"' + name + '.num_repeat_trials" must be an integer in [0, ' + name + '.num_experimental_trials] (' + stage.num_experimental_trials + '), got ' + stage.num_repeat_trials + '.');
        if (!Number.isInteger(stage.num_catch_trials) || stage.num_catch_trials < 0)
            err('"' + name + '.num_catch_trials" must be a non-negative integer, got ' + stage.num_catch_trials + '.');
        if (!Number.isInteger(stage.min_repeat_separation) || stage.min_repeat_separation < 0)
            err('"' + name + '.min_repeat_separation" must be a non-negative integer, got ' + stage.min_repeat_separation + '.');
    }

    // 2b. screening_block.thresholds ranges (each may be null — skip range check if so)
    const thr = sb.thresholds;
    if (thr.min_reliability !== null && (thr.min_reliability < -1 || thr.min_reliability > 1))
        err('"screening_block.thresholds.min_reliability" must be in [-1,1] or null, got ' + thr.min_reliability + '.');
    if (thr.median_reliability !== null && (thr.median_reliability < -1 || thr.median_reliability > 1))
        err('"screening_block.thresholds.median_reliability" must be in [-1,1] or null, got ' + thr.median_reliability + '.');
    if (thr.move_ratio_max_fail_rate !== null && (thr.move_ratio_max_fail_rate < 0 || thr.move_ratio_max_fail_rate > 1))
        err('"screening_block.thresholds.move_ratio_max_fail_rate" must be in [0,1] or null, got ' + thr.move_ratio_max_fail_rate + '.');
    if (thr.distance_sd_max_fail_rate !== null && (thr.distance_sd_max_fail_rate < 0 || thr.distance_sd_max_fail_rate > 1))
        err('"screening_block.thresholds.distance_sd_max_fail_rate" must be in [0,1] or null, got ' + thr.distance_sd_max_fail_rate + '.');

    if (minW <= 0) err('"display.sort_area_min_width" must be > 0, got '    + minW + '.');
    if (minH <= 0) err('"display.sort_area_min_height" must be > 0, got '   + minH + '.');
    if (minHeader < 0) err('"display.min_header_height_px" must be >= 0, got ' + minHeader + '.');
    if (minFooter < 0) err('"display.min_footer_height_px" must be >= 0, got ' + minFooter + '.');
    if (frac  <= 0 || frac  >= 1) err('"display.image_size_fraction" must be in (0, 1), got ' + frac  + '.');
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

    // 3a. min_repeat_separation structural feasibility, checked INDEPENDENTLY
    // for each stage (they can have different trial/repeat counts, unlike
    // the old shared-uniform-block model where one check sufficed for all
    // blocks). Slot positions are fully determined by num_experimental_trials
    // and num_repeat_trials alone — independent of the RNG — so this can be
    // checked at config-load time. Failing here means insertTrialRepeats is
    // guaranteed to fail too. screening_block is skipped entirely if disabled.
    if (sb.enabled) {
        const tScreen = sb.num_experimental_trials + sb.num_repeat_trials;
        if (!canSatisfyTrialRepeatSeparation(tScreen, sb.num_repeat_trials, sb.min_repeat_separation))
            err('screening_block.min_repeat_separation (' + sb.min_repeat_separation + ') cannot be satisfied within ' +
                'screening_block (num_experimental_trials=' + sb.num_experimental_trials + ' + num_repeat_trials=' +
                sb.num_repeat_trials + ' = ' + tScreen + ' slots). Lower min_repeat_separation, lower num_repeat_trials, ' +
                'or increase num_experimental_trials. Note: catch trials are not counted toward this separation.');
    }
    const tExp = eb.num_experimental_trials + eb.num_repeat_trials;
    if (!canSatisfyTrialRepeatSeparation(tExp, eb.num_repeat_trials, eb.min_repeat_separation))
        err('experimental_block.min_repeat_separation (' + eb.min_repeat_separation + ') cannot be satisfied within ' +
            'experimental_block (num_experimental_trials=' + eb.num_experimental_trials + ' + num_repeat_trials=' +
            eb.num_repeat_trials + ' = ' + tExp + ' slots). Lower min_repeat_separation, lower num_repeat_trials, ' +
            'or increase num_experimental_trials. Note: catch trials are not counted toward this separation.');

    // 3c. Single image fits in minimum sort area
    const stimSize = Math.round(minW * frac);
    if (stimSize >= minW) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds display.sort_area_min_width ('  + minW + 'px). Reduce display.image_size_fraction.');
    if (stimSize >= minH) err('Computed stimulus size (' + stimSize + 'px) equals or exceeds display.sort_area_min_height (' + minH + 'px). Reduce display.image_size_fraction.');

    // 3d. k images fit in a square grid within the minimum sort area
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
        if (!config.experimental_trials.stimuli_path) warn('"experimental_trials.stimuli_path" is empty — no main images will load.');
        if (!config.catch_trials.stimuli_path)         warn('"catch_trials.stimuli_path" is empty — catch and practice trials will have no images.');
        if (config.screening_block.enabled && !config.screening_block.prolific_code)
            warn('"screening_block.prolific_code" is empty — participants screened out will not be redirected.');
        if (!config.experimental_block.prolific_code)
            warn('"experimental_block.prolific_code" is empty — participants who complete the experiment will not be redirected.');
        if (!config.consent.no_consent_code)
            warn('"consent.no_consent_code" is empty — participants who decline consent will not be redirected.');
    }
}

// Images always start at random positions inside the sort area (SpAM convention;
// see CLAUDE.md "Trial layout"). Staging-column placement (images starting outside
// the arena) is not supported.
const STIM_STARTS_INSIDE = true;

// Fraction of viewport width the sort area fills (floor-protected by
// display.sort_area_min_width).
const WIDTH_FILL_FRACTION = 0.97;

/**
 * Compute the sort-area dimensions and stimulus size for the current viewport.
 *
 * The sort area fills WIDTH_FILL_FRACTION of viewport width, and all remaining
 * viewport height after reserving display.min_header_height_px (prompt strip, above
 * the arena) and display.min_footer_height_px (counter + Done button strip, below
 * the arena) — there's no DOM-measurement infrastructure here, computeLayout runs
 * once at page load before any trial DOM exists, so these are fixed pixel budgets,
 * not measured. Both axes are floor-protected by sort_area_min_width/height for
 * small screens. Image size is expressed as a fraction of the sort-area width so
 * that emoji and dataset images are always rendered at the same size regardless of
 * their native resolution. The plugin's arena border is nested inside the outer
 * arena div at 94% size (centered), so it's fully contained within sortH -- no
 * extra pixel budget needed for the border itself.
 *
 * @param {number} viewportW - window.innerWidth
 * @param {number} viewportH - window.innerHeight
 * @param {{ display: {
 *   sort_area_min_width: number, sort_area_min_height: number,
 *   min_header_height_px: number, min_footer_height_px: number,
 *   image_size_fraction: number,
 * } }} config
 * @returns {{ sortW: number, sortH: number, stimSize: number }}
 */
function computeLayout(viewportW, viewportH, config) {
    const {
        sort_area_min_width, sort_area_min_height,
        min_header_height_px, min_footer_height_px,
        image_size_fraction,
    } = config.display;

    const sortW = Math.max(
        sort_area_min_width,
        Math.floor(viewportW * WIDTH_FILL_FRACTION),
    );
    const sortH = Math.max(
        sort_area_min_height,
        viewportH - min_header_height_px - min_footer_height_px,
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
            const distance = roundTo(Math.sqrt(dx * dx + dy * dy) / diagonal);
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
 * @param {object} config      - Task config (reads experimental_trials.min_pairwise_distance_sd / experimental_trials.min_move_item_ratio)
 * @returns {boolean} true if the trial should be flagged
 */
function computeMainQcFlag(sd, numMoves, numItems, config) {
    const et = config.experimental_trials;
    const enoughMoves = numMoves >= et.min_move_item_ratio * numItems;
    return sd < et.min_pairwise_distance_sd || !enoughMoves;
}

/**
 * Spearman rank correlation between two pairwise-distance vectors, matched
 * by UNORDERED image-pair identity {src1,src2} (not array order —
 * computePairwiseDistances' output order depends on final_locations' order,
 * which differs between an original trial and its verbatim repeat because
 * insertTrialRepeats reshuffles the repeat's presentation order).
 *
 * @param {Array<{src1: string, src2: string, distance: number}>} pairsA - original trial's pairwise distances
 * @param {Array<{src1: string, src2: string, distance: number}>} pairsB - repeat trial's pairwise distances
 * @returns {number} Spearman rho in [-1,1], or NaN if either vector has <2 pairs or zero variance
 * @throws {Error} if pairsA/pairsB don't cover exactly the same set of unordered pairs
 */
function computeSpearmanCorrelation(pairsA, pairsB) {
    // NUL separator: it cannot occur inside an image URL, so no filename can
    // forge another pair's key. Written as an escape sequence, not a literal
    // NUL byte -- a literal makes git and grep treat this file as binary.
    const keyOf = p => [p.src1, p.src2].sort().join('\u0000');
    const mapA  = new Map(pairsA.map(p => [keyOf(p), p.distance]));
    const mapB  = new Map(pairsB.map(p => [keyOf(p), p.distance]));
    if (mapA.size !== pairsA.length) throw new Error('computeSpearmanCorrelation: pairsA has duplicate pair keys.');
    if (mapB.size !== pairsB.length) throw new Error('computeSpearmanCorrelation: pairsB has duplicate pair keys.');
    if (mapA.size !== mapB.size)
        throw new Error(`computeSpearmanCorrelation: pairsA has ${mapA.size} unique pairs, pairsB has ${mapB.size} — image sets differ.`);

    const valuesA = [], valuesB = [];
    for (const [key, distA] of mapA) {
        if (!mapB.has(key)) throw new Error(`computeSpearmanCorrelation: pair "${key}" present in pairsA but not pairsB.`);
        valuesA.push(distA);
        valuesB.push(mapB.get(key));
    }
    return roundTo(_pearsonCorrelation(_averageRanks(valuesA), _averageRanks(valuesB)));
}

/** Rank values 1..n, tied values get the AVERAGE of their tied-block ranks. */
function _averageRanks(values) {
    const n   = values.length;
    const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    const ranks = new Array(n);
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && values[idx[j + 1]] === values[idx[i]]) j++;
        const avgRank = (i + j) / 2 + 1; // 1-based
        for (let m = i; m <= j; m++) ranks[idx[m]] = avgRank;
        i = j + 1;
    }
    return ranks;
}

function _pearsonCorrelation(a, b) {
    const n = a.length;
    if (n < 2) return NaN;
    const meanA = a.reduce((s, v) => s + v, 0) / n;
    const meanB = b.reduce((s, v) => s + v, 0) / n;
    let num = 0, denA = 0, denB = 0;
    for (let i = 0; i < n; i++) {
        const da = a[i] - meanA, db = b[i] - meanB;
        num += da * db; denA += da * da; denB += db * db;
    }
    if (denA === 0 || denB === 0) return NaN;
    return num / Math.sqrt(denA * denB);
}

/**
 * Evaluate cumulative screening criteria at the end of the screening stage.
 * Pure function over already-computed per-trial stats — no DOM/jsPsych
 * dependency, so it is unit-testable in isolation.
 *
 * moveRatioFailRate/distanceSdFailRate are deliberately independent counts
 * against independent thresholds — NOT computeMainQcFlag's OR-combined
 * boolean, which conflates the two failure modes. computeMainQcFlag remains
 * the sole driver of the per-trial qc_flag CSV column; this function is a
 * separate, cumulative aggregation for the one-time screening evaluation.
 *
 * Each of the 4 threshold criteria is independently skipped for two reasons:
 * (1) its config value is `null` ("always disabled"), or (2) — for the two
 * reliability criteria only — zero repeats have completed yet, so there is
 * no data to evaluate regardless of the threshold's value.
 *
 * @param {{mainTrials: Array<{numMoves: number, numItems: number, sd: number}>,
 *          reliabilities: number[]}} dataSoFar
 *   mainTrials: one entry per completed main trial (both distinct and repeat)
 *   in the screening stage so far. reliabilities: one Spearman R per
 *   completed repeat trial so far.
 * @param {{screening_block: {thresholds: {min_reliability: (number|null), median_reliability: (number|null),
 *                      move_ratio_max_fail_rate: (number|null), distance_sd_max_fail_rate: (number|null)}},
 *          experimental_trials: {min_move_item_ratio: number, min_pairwise_distance_sd: number}}} config
 * @returns {{pass: boolean, reasons: string[],
 *            stats: {moveRatioFailRate: number, distanceSdFailRate: number,
 *                    minReliability: (number|null), medianReliability: (number|null)}}}
 */
function evaluateScreening(dataSoFar, config) {
    const { mainTrials, reliabilities } = dataSoFar;
    const thr = config.screening_block.thresholds;
    const et  = config.experimental_trials;

    const n = mainTrials.length;
    const moveFails = mainTrials.filter(t => t.numMoves < et.min_move_item_ratio * t.numItems).length;
    const sdFails    = mainTrials.filter(t => t.sd < et.min_pairwise_distance_sd).length;
    const moveRatioFailRate  = roundTo(n === 0 ? 0 : moveFails / n);
    const distanceSdFailRate = roundTo(n === 0 ? 0 : sdFails   / n);
    const minReliability    = reliabilities.length === 0 ? null : roundTo(Math.min(...reliabilities));
    const medianReliability = reliabilities.length === 0 ? null : roundTo(_median(reliabilities));

    const reasons = [];
    if (thr.move_ratio_max_fail_rate !== null && moveRatioFailRate > thr.move_ratio_max_fail_rate)
        reasons.push(`move-ratio fail rate ${moveRatioFailRate.toFixed(3)} exceeds move_ratio_max_fail_rate (${thr.move_ratio_max_fail_rate})`);
    if (thr.distance_sd_max_fail_rate !== null && distanceSdFailRate > thr.distance_sd_max_fail_rate)
        reasons.push(`distance-SD fail rate ${distanceSdFailRate.toFixed(3)} exceeds distance_sd_max_fail_rate (${thr.distance_sd_max_fail_rate})`);
    if (thr.min_reliability !== null && minReliability !== null && minReliability < thr.min_reliability)
        reasons.push(`minimum reliability ${minReliability.toFixed(3)} is below min_reliability (${thr.min_reliability})`);
    if (thr.median_reliability !== null && medianReliability !== null && medianReliability < thr.median_reliability)
        reasons.push(`median reliability ${medianReliability.toFixed(3)} is below median_reliability (${thr.median_reliability})`);

    return { pass: reasons.length === 0, reasons, stats: { moveRatioFailRate, distanceSdFailRate, minReliability, medianReliability } };
}

/** Median of a numeric array (average of the two middle values for even length). */
function _median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}
