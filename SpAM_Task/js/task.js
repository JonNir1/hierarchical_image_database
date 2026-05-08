'use strict';

// Requires (loaded via <script> tags before this file):
//   jspsych.js, plugin-preload.js, plugin-html-button-response.js,
//   plugin-fullscreen.js, plugin-free-sort.js, seedrandom.js,
//   jspsych-7-pavlovia-2022.1.1.js, utils.js, trial_generator.js

document.addEventListener('DOMContentLoaded', async () => {

  // ---------------------------------------------------------------------------
  // 1. Load config and manifest
  // ---------------------------------------------------------------------------
  let config, manifest;
  try {
    const [configRes, manifestRes] = await Promise.all([
      fetch('config.json'),
      fetch('stimuli_manifest.json'),
    ]);
    config   = await configRes.json();
    manifest = await manifestRes.json();
  } catch (err) {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      'Failed to load experiment configuration. Please contact the researcher.<br><small>' +
      err.message + '</small></p>';
    return;
  }

  // Validate config before any computation.
  try {
    verifyConfig(config);
  } catch (err) {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      '<strong>Configuration error:</strong><br>' + err.message + '</p>';
    return;
  }

  // ---------------------------------------------------------------------------
  // 2. Environment + participant ID
  // ---------------------------------------------------------------------------
  const isPavlovia = window.location.hostname.includes('pavlovia.org');

  const params = new URLSearchParams(window.location.search);
  const rawPid = params.get('PROLIFIC_PID');

  if (!rawPid && !config.debug) {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      'No participant ID detected. Please access this experiment via your Prolific invitation link.</p>';
    return;
  }

  const PID = config.debug ? 'debug_participant' : rawPid;
  if (config.debug) console.log('[SpAM] Debug mode active. PID:', PID);

  // ---------------------------------------------------------------------------
  // 3. Seeded RNG
  // ---------------------------------------------------------------------------
  const seed = hashString(PID);
  // rng must not be called before buildTrialLists — any prior call shifts the
  // sequence and breaks reproducibility for the same PID.
  const rng  = new Math.seedrandom(seed);
  if (config.debug) console.log('[SpAM] Seed:', seed);

  // ---------------------------------------------------------------------------
  // 4. Build image URL arrays
  //    Manifest paths are filenames relative to each stimulus directory.
  //    config.*_path values are relative to SpAM_Task/ (same dir as index.html),
  //    so concatenating them gives browser-resolvable relative URLs.
  // TODO: if config paths are authored on Windows, replace backslashes:
  //       config.stimuli_path.replace(/\\/g, '/')
  // ---------------------------------------------------------------------------
  const imageUrls    = (manifest.images          || []).map(f => config.stimuli_path          + '/' + f);
  const catchUrls    = (manifest.catch_images    || []).map(f => config.stimuli_catch_path    + '/' + f);
  const practiceUrls = (manifest.practice_images || []).map(f => config.stimuli_practice_path + '/' + f);

  // ---------------------------------------------------------------------------
  // 5. Responsive layout — computed once from viewport at page-load time
  // ---------------------------------------------------------------------------
  const { sortW, sortH, stimSize } = computeLayout(
    window.innerWidth, window.innerHeight, config,
  );
  if (config.debug) console.log('[SpAM] Layout:', { sortW, sortH, stimSize });

  // jsPsychFreeSort only recognises 'square' (rectangular) and 'ellipse'.
  // Our config uses 'rect'/'ellipse'; map here so the inside-check uses the
  // correct geometry (passing an unrecognised value falls through to ellipse,
  // which would exclude images placed in the corners of a rectangular arena).
  const pluginShape = config.sort_area_shape === 'ellipse' ? 'ellipse' : 'square';

  // ---------------------------------------------------------------------------
  // 6. Trial lists
  //    buildTrialLists expects rng as a function (not a seed integer).
  //    rng must not be called between here and buildTrialLists — any prior call
  //    shifts the sequence and breaks per-PID reproducibility.
  //    insertCatchTrials uses the same rng instance (continued sequence) so
  //    catch location assignment is also seeded and recorded in trial data.
  // ---------------------------------------------------------------------------
  const mainTrials = buildTrialLists(imageUrls, config, rng);
  const allTrials  = insertCatchTrials(mainTrials, catchUrls, config, rng);

  if (config.debug) {
    console.log('[SpAM] Trial sequence:', allTrials.map((t, i) => i + ':' + t.type));
  }

  // ---------------------------------------------------------------------------
  // 7. jsPsych initialisation
  // ---------------------------------------------------------------------------
  const jsPsych = initJsPsych({
    auto_preload: false, // we preload per-trial via jsPsychPreload nodes
    on_finish:    saveData,
  });

  // ---------------------------------------------------------------------------
  // 8. saveData — called by jsPsych.on_finish
  //    On Pavlovia: the jsPsychPavlovia finish node (in the timeline below)
  //    handles saving before on_finish fires; nothing to do here.
  //    Locally: download a filtered CSV (practice trials excluded).
  // ---------------------------------------------------------------------------
  /**
   * Save experiment data at the end of the session.
   *
   * On Pavlovia: the `jsPsychPavlovia` finish node in the timeline handles upload
   * before `on_finish` fires, so this function is a no-op in that environment.
   *
   * Locally: filters jsPsych data to main and catch trials (practice excluded),
   * serialises to CSV, and triggers a browser download via a temporary <a> element.
   * Filename format: `spam_<PID>_<timestamp>.csv`.
   */
  function saveData() {
    if (isPavlovia) return;

    const csv  = jsPsych.data.get()
                   .filter(d => d.trial_type === 'main' || d.trial_type === 'catch')
                   .csv();
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'spam_' + PID + '_' + Date.now() + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---------------------------------------------------------------------------
  // 9. Timeline
  // ---------------------------------------------------------------------------
  const timeline = [];

  // --- Pavlovia init (Pavlovia only) ---
  if (isPavlovia) {
    timeline.push({ type: jsPsychPavlovia, command: 'init' });
  }

  // --- Consent ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:700px; text-align:left;">
        <h2>Participant Information &amp; Consent</h2>
        <p>In this study you will arrange images of objects according to how visually
           similar they appear to you. The task takes approximately 20–30 minutes.</p>
        <p>Your participation is voluntary. Responses are recorded anonymously and will
           be used for academic research only. You may withdraw at any time by closing
           the browser tab.</p>
        <p>By clicking <strong>I agree to participate</strong> you confirm that you are
           18 years of age or older and consent to take part.</p>
      </div>`,
    choices: ['I agree to participate'],
  });

  // --- Instructions ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:700px; text-align:left;">
        <h2>Instructions</h2>
        <p>On each trial you will see a sorting area containing several object images.
           Drag them to positions that reflect their <strong>visual similarity</strong>:</p>
        <ul>
          <li><strong>Similar-looking</strong> images → place them <strong>close together</strong>.</li>
          <li><strong>Different-looking</strong> images → place them <strong>far apart</strong>.</li>
        </ul>
        <p>There are no right or wrong answers — go with your first impression.</p>
        <p>You will first do a short <strong>practice trial</strong> (not recorded),
           then ${allTrials.length} real trials.</p>
        <p>Please work in <strong>fullscreen</strong>. Do not use the back button.</p>
      </div>`,
    choices: ['Start practice'],
  });

  // --- Enter fullscreen ---
  timeline.push({
    type:           jsPsychFullscreen,
    fullscreen_mode: true,
    message:        '<p>The task will now switch to fullscreen mode.</p>',
    button_label:   'Continue',
  });

  // --- Practice trial ---
  timeline.push({
    type:            jsPsychFreeSort,
    stimuli:         practiceUrls.slice(0, config.practice_images_per_trial),
    sort_area_width:  sortW,
    sort_area_height: sortH,
    stim_width:        stimSize,
    stim_height:       stimSize,
    sort_area_shape:      pluginShape,
    stim_starts_inside:   config.stim_starts_inside,
    column_spread_factor: config.column_spread_factor,
    prompt: '<p style="font-size:0.9em; color:#333;">PRACTICE — arrange these images by visual similarity. ' +
            'This trial will <strong>not</strong> be recorded.</p>',
    on_finish(data) {
      data.trial_type = 'practice';
    },
  });

  // --- Post-practice transition ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: '<p>Practice complete. The real trials begin now.</p>' +
              '<p>Remember: close together = similar, far apart = different.</p>',
    choices:  ['Begin'],
  });

  // --- Main + catch trials ---
  allTrials.forEach((trial, idx) => {

    // Per-trial preload (only this trial's images)
    timeline.push({
      type:    jsPsychPreload,
      images:  trial.images,
      message: '<p>Loading images…</p>',
      show_progress_bar: true,
      continue_after_error: true,
      error_message: '<p>One or more images failed to load. Please contact the researcher.</p>',
    });

    // Free-sort trial
    timeline.push({
      type:             jsPsychFreeSort,
      stimuli:          trial.images,
      sort_area_width:  sortW,
      sort_area_height: sortH,
      stim_width:        stimSize,
      stim_height:       stimSize,
      sort_area_shape:      pluginShape,
      stim_starts_inside:   config.stim_starts_inside,
      column_spread_factor: config.column_spread_factor,
      prompt: trial.type === 'catch'
        ? '<p style="font-size:0.9em; color:#333;">Please place all images on the ' +
          '<strong>' + trial.target_location + '</strong> of the screen.</p>'
        : '<p style="font-size:0.9em; color:#333;">Arrange the images by visual similarity. ' +
          'Close together = similar &nbsp;|&nbsp; Far apart = different.</p>',
      on_finish(data) {
        // QC metrics
        const pairs = computePairwiseDistances(
          data.final_locations,
          sortW,
          sortH,
        );
        const distances = pairs.map(p => p.distance);
        const sd = computeSD(distances);

        data.trial_type            = trial.type; // 'main' or 'catch'
        data.trial_index           = idx;
        data.pairwise_distance_sd  = sd;

        if (trial.type === 'catch') {
          // Catch-trial QC: cluster tightness + centroid proximity to target.
          // computeCentroid and isCentroidNearTarget added with new catch design.
          const centroid   = computeCentroid(data.final_locations);
          const clusterMean = distances.reduce((a, b) => a + b, 0) / (distances.length || 1);
          const locationOk = isCentroidNearTarget(
            centroid, trial.target_location,
            sortW, sortH,
            config.catch_location_tolerance,
          );
          data.target_location        = trial.target_location;
          data.centroid_x             = centroid.x;
          data.centroid_y             = centroid.y;
          data.cluster_mean_distance  = clusterMean;
          data.qc_flag = clusterMean > config.catch_cluster_max_mean ||
                         sd          > config.catch_cluster_max_sd   ||
                         !locationOk ||
                         data.rt < config.min_trial_rt_ms;
        } else {
          data.qc_flag = sd < config.min_pairwise_distance_sd ||
                         data.rt < config.min_trial_rt_ms;
        }

        // Store pairs as JSON string for CSV compatibility
        data.pairwise_distances = JSON.stringify(pairs);

        if (config.debug) {
          console.log(
            '[SpAM] Trial', idx, '(' + trial.type + ')',
            '| sd=' + sd.toFixed(4),
            '| rt=' + data.rt + 'ms',
            '| qc_flag=' + data.qc_flag,
          );
        }
      },
    });
  });

  // --- Debrief ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:600px; text-align:center;">
        <h2>Thank you!</h2>
        <p>You have completed all trials. Your responses are being saved.</p>
        <p>Click <strong>Finish</strong> to return to Prolific and receive your completion credit.</p>
      </div>`,
    choices: ['Finish'],
  });

  // --- Pavlovia finish (Pavlovia only) ---
  if (isPavlovia) {
    timeline.push({
      type:    jsPsychPavlovia,
      command: 'finish',
      completedCallback() {
        if (config.prolific_completion_url) {
          window.location.href = config.prolific_completion_url;
        }
      },
    });
  }

  // ---------------------------------------------------------------------------
  // 10. Run
  // ---------------------------------------------------------------------------
  jsPsych.run(timeline);

}); // end DOMContentLoaded
