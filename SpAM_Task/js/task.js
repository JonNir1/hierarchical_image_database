'use strict';

// Requires (loaded via <script> tags before this file):
//   jspsych.js, plugin-preload.js, plugin-html-button-response.js,
//   plugin-fullscreen.js, plugin-free-sort-patched.js, seedrandom.js,
//   jspsych-7-pavlovia-2022.1.1.js, utils.js, trial_generator.js

document.addEventListener('DOMContentLoaded', async () => {

  // ---------------------------------------------------------------------------
  // 1. Load config and manifest
  // ---------------------------------------------------------------------------
  let config, manifest;
  try {
    const [configRes, manifestRes] = await Promise.all([
      fetch('task_config.json'),
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

  if (!rawPid && !config.deployment.debug) {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      'No participant ID detected. Please access this experiment via your Prolific invitation link.</p>';
    return;
  }

  const PID = config.deployment.debug ? 'debug_participant' : rawPid;
  if (config.deployment.debug) console.log('[SpAM] Debug mode active. PID:', PID);

  // ---------------------------------------------------------------------------
  // 3. Seeded RNG
  // ---------------------------------------------------------------------------
  const seed = hashString(PID);
  // rng must not be called before buildTrialLists — any prior call shifts the
  // sequence and breaks reproducibility for the same PID.
  const rng  = new Math.seedrandom(seed);
  if (config.deployment.debug) console.log('[SpAM] Seed:', seed);

  // ---------------------------------------------------------------------------
  // 4. SHINE variant — cross-check manifest against config
  //    generate_manifest.py writes the active variant into the manifest at build
  //    time. If the config has been edited since the manifest was generated, the
  //    two will disagree and the wrong images would load. Fail loudly instead.
  // ---------------------------------------------------------------------------
  const variant = config.shine.shine_variant;
  if (manifest.shine_variant !== variant) {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      '<strong>Configuration / manifest mismatch:</strong><br>' +
      'config.shine.shine_variant = "' + variant + '" but stimuli_manifest.json was built ' +
      'for "' + manifest.shine_variant + '".<br>Re-run generate_manifest.py and reload.</p>';
    return;
  }
  if (config.deployment.debug) console.log('[SpAM] SHINE variant:', variant);

  // ---------------------------------------------------------------------------
  // 5. Build image URL arrays
  //    Manifest paths are filenames relative to each stimulus directory.
  //    Main URL = stimuli_paths.main_root + "<variant>_shine/" + filename;
  //    practice/catch URLs are unchanged.
  //    All path values are relative to SpAM_Task/ (same dir as index.html),
  //    so concatenating them gives browser-resolvable relative URLs.
  // TODO: if config paths are authored on Windows, replace backslashes:
  //       config.stimuli_path.replace(/\\/g, '/')
  // ---------------------------------------------------------------------------
  const mainPrefix   = config.stimuli_paths.main_root + variant + '_shine';
  const imageUrls    = (manifest.images          || []).map(f => mainPrefix                   + '/' + f);
  const catchUrls    = (manifest.catch_images    || []).map(f => config.stimuli_paths.catch    + '/' + f);
  const practiceUrls = (manifest.practice_images || []).map(f => config.stimuli_paths.practice + '/' + f);

  // ---------------------------------------------------------------------------
  // 5. Responsive layout — computed once from viewport at page-load time
  // ---------------------------------------------------------------------------
  const { sortW, sortH, stimSize } = computeLayout(
    window.innerWidth, window.innerHeight, config,
  );
  if (config.deployment.debug) console.log('[SpAM] Layout:', { sortW, sortH, stimSize });

  // jsPsychFreeSort only recognises 'square' (rectangular) and 'ellipse'.
  // Our config uses 'rect'/'ellipse'; map here so the inside-check uses the
  // correct geometry (passing an unrecognised value falls through to ellipse,
  // which would exclude images placed in the corners of a rectangular arena).
  const pluginShape = config.display.sort_area_shape === 'ellipse' ? 'ellipse' : 'square';

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

  if (config.deployment.debug) {
    console.log('[SpAM] Trial sequence:', allTrials.map((t, i) => i + ':' + t.type));
  }

  // ---------------------------------------------------------------------------
  // 7. jsPsych initialisation
  // ---------------------------------------------------------------------------
  const jsPsych = initJsPsych({
    auto_preload: false, // we preload per-trial via jsPsychPreload nodes
    on_finish:    onFinish,
  });

  // ---------------------------------------------------------------------------
  // 8. onFinish / saveData — called by jsPsych.on_finish
  //    On Pavlovia: the jsPsychPavlovia finish node (in the timeline below)
  //    handles saving before on_finish fires; nothing to do here except show
  //    the end screen.
  //    Locally: download a filtered CSV (practice trials excluded), then show
  //    the end screen.
  // ---------------------------------------------------------------------------
  function showEndScreen() {
    document.body.innerHTML = `
      <div style="
        display: flex; align-items: center; justify-content: center;
        height: 100vh; color: #ddd; font-family: sans-serif; text-align: center;
      ">
        <div>
          <p>Press <strong>Esc</strong> to exit full screen.</p>
          <p>You may now close this window.</p>
        </div>
      </div>`;
  }

  /**
   * Save experiment data at the end of the session, then show the end screen.
   *
   * On Pavlovia: the `jsPsychPavlovia` finish node in the timeline handles upload
   * before `on_finish` fires, so only the end screen is shown here.
   *
   * Locally: filters jsPsych data to main and catch trials (practice excluded),
   * serialises to CSV, and triggers a browser download via a temporary <a> element.
   * Filename format: `spam_<PID>_<timestamp>.csv`.
   */
  function onFinish() {
    if (!isPavlovia) {
      const csv  = jsPsych.data.get()
                     .filterCustom(d => d.trial_type.startsWith('trial_') ||
                                        d.trial_type.startsWith('catch_'))
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
    showEndScreen();
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
           similar they appear to you. The task takes approximately 45-60 minutes.</p>
        <p>Your participation is voluntary. Responses are recorded anonymously and will
           be used for solely academic research only. You may withdraw at any time by
           closing the browser tab.</p>
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
        <p>You will first do a short <strong>practice trial</strong>,
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
    stimuli:         practiceUrls.slice(0, config.design.practice_images_per_trial),
    sort_area_width:  sortW,
    sort_area_height: sortH,
    stim_width:        stimSize,
    stim_height:       stimSize,
    sort_area_shape:      pluginShape,
    stim_starts_inside:   config.display.stim_starts_inside,
    column_spread_factor: config.display.column_spread_factor,
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
  let mainCount = 0, catchCount = 0;
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
    let header_text = trial.type === 'catch'
        ? '<p style="font-size:0.9em; color:#333;">Please place all images on the ' +
          '<strong>' + trial.target_location + '</strong> of the screen.</p>'
        : '<p style="font-size:0.9em; color:#333;">Arrange the images by visual similarity. ' +
          'Close together = similar &nbsp;|&nbsp; Far apart = different.</p>';
    timeline.push({
      type:             jsPsychFreeSort,
      stimuli:          trial.images,
      sort_area_width:  sortW,
      sort_area_height: sortH,
      stim_width:        stimSize,
      stim_height:       stimSize,
      sort_area_shape:      pluginShape,
      stim_starts_inside:   config.display.stim_starts_inside,
      column_spread_factor: config.display.column_spread_factor,
      prompt: '',
      counter_text_unfinished:  header_text,
      counter_text_finished:    header_text,
      on_finish(data) {
        // QC metrics
        const pairs = computePairwiseDistances(
          data.final_locations,
          sortW,
          sortH,
        );
        const distances = pairs.map(p => p.distance);
        const sd = computeSD(distances);

        // trial_type: 'trial_N' for main trials, 'catch_N' for catch trials
        // (1-based running index within each category)
        data.trial_type  = trial.type === 'catch'
          ? 'catch_' + (++catchCount)
          : 'trial_' + (++mainCount);
        data.trial_index           = idx;
        data.shine_variant         = variant;
        data.pairwise_distance_sd  = sd;

        if (trial.type === 'catch') {
          // Catch-trial QC: cluster tightness + centroid proximity to target.
          // computeCentroid and isCentroidNearTarget added with new catch design.
          const centroid   = computeCentroid(data.final_locations);
          const clusterMean = distances.reduce((a, b) => a + b, 0) / (distances.length || 1);
          const locationOk = isCentroidNearTarget(
            centroid, trial.target_location,
            sortW, sortH,
            config.catch_trials.location_tolerance,
          );
          data.target_location        = trial.target_location;
          data.centroid_x             = centroid.x;
          data.centroid_y             = centroid.y;
          data.cluster_mean_distance  = clusterMean;
          data.qc_flag = clusterMean > config.catch_trials.cluster_max_mean  ||
                         sd          > config.catch_trials.cluster_max_sd    ||
                         !locationOk ||
                         data.rt < config.quality_control.min_trial_rt_ms;
        } else {
          data.qc_flag = sd < config.quality_control.min_pairwise_distance_sd ||
                         data.rt < config.quality_control.min_trial_rt_ms;
        }

        // Store pairs as JSON string for CSV compatibility
        data.pairwise_distances = JSON.stringify(pairs);

        if (config.deployment.debug) {
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
        if (config.deployment.prolific_completion_url) {
          window.location.href = config.deployment.prolific_completion_url;
        }
      },
    });
  }

  // ---------------------------------------------------------------------------
  // 10. Run
  // ---------------------------------------------------------------------------
  jsPsych.run(timeline);

}); // end DOMContentLoaded
