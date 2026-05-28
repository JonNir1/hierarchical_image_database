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
      fetch('SpAM_Task/task_config.json'),
      fetch('SpAM_Task/stimuli_manifest.json'),
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

  // Apply global typography and background from config.display.
  // Setting these on <body> lets all jsPsych screens inherit them without
  // needing per-element inline styles.
  (function applyGlobalStyles(disp) {
    const style = document.createElement('style');
    style.textContent = [
      'body {',
      '  background-color: ' + disp.background_color + ';',
      '  color: '            + disp.text_color        + ';',
      '  font-family: '      + disp.font_family        + ';',
      '  font-size: '        + disp.font_size          + ';',
      '  line-height: '      + disp.line_height        + ';',
      '}',
    ].join('\n');
    document.head.appendChild(style);
  }(config.display));

  // ---------------------------------------------------------------------------
  // 2. Environment + participant ID
  // ---------------------------------------------------------------------------
  const isPavlovia = window.location.hostname.includes('pavlovia.org');
  const mode = config.deployment.mode;

  const params = new URLSearchParams(window.location.search);
  const rawPid = params.get('PROLIFIC_PID');

  if (!rawPid && mode !== 'debug') {
    document.body.innerHTML =
      '<p style="color:white;font-family:sans-serif;text-align:center;margin-top:20%;">' +
      'No participant ID detected. Please access this experiment via your Prolific invitation link.</p>';
    return;
  }

  const PID = mode === 'debug' ? 'debug_participant' : rawPid;
  if (mode === 'debug') console.log('[SpAM] Debug mode active. PID:', PID);

  // ---------------------------------------------------------------------------
  // 3. Seeded RNG
  // ---------------------------------------------------------------------------
  const seed = hashString(PID);
  // rng must not be called before buildTrialLists — any prior call shifts the
  // sequence and breaks reproducibility for the same PID.
  const rng  = new Math.seedrandom(seed);
  if (mode === 'debug') console.log('[SpAM] Seed:', seed);

  // ---------------------------------------------------------------------------
  // 4. Cohort assignment
  //    debug  → use config.deployment.debug_shine_variant
  //    pilot  → always "pre" (pre-SHINE cohort for pilot runs)
  //    production → deterministic from PID hash; even → "pre", odd → "post"
  //    hashString(PID) % 2 uses the hash value directly — it does NOT consume
  //    from the rng instance, so trial reproducibility is unaffected.
  // ---------------------------------------------------------------------------
  let assignedVariant;
  if (mode === 'debug') {
    assignedVariant = config.deployment.debug_shine_variant;
  } else if (mode === 'pilot') {
    assignedVariant = 'pre';
  } else {
    assignedVariant = hashString(PID) % 2 === 0 ? 'pre' : 'post';
  }
  if (mode === 'debug') console.log('[SpAM] SHINE variant:', assignedVariant, '(mode:', mode + ')');

  // ---------------------------------------------------------------------------
  // 5. Build image URL arrays
  //    Manifest paths are filenames relative to each stimulus directory.
  //    Main URL = stimuli_paths.main_root + "<variant>_shine/" + filename;
  //    practice/catch URLs are unchanged.
  //    All stimuli_paths values are relative to the repo root (where
  //    index.html lives), so concatenating them with filenames yields
  //    browser-resolvable URLs.
  // ---------------------------------------------------------------------------
  const mainPrefix   = config.stimuli_paths.main_root + assignedVariant + '_shine';
  const imageUrls    = (manifest.images          || []).map(f => mainPrefix                   + '/' + f);
  const catchUrls    = (manifest.catch_images    || []).map(f => config.stimuli_paths.catch    + '/' + f);
  const practiceUrls = (manifest.practice_images || []).map(f => config.stimuli_paths.practice + '/' + f);

  // ---------------------------------------------------------------------------
  // 5. Responsive layout — computed once from viewport at page-load time
  // ---------------------------------------------------------------------------
  const { sortW, sortH, stimSize } = computeLayout(
    window.innerWidth, window.innerHeight, config,
  );
  if (mode === 'debug') console.log('[SpAM] Layout:', { sortW, sortH, stimSize });

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

  if (mode === 'debug') {
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
        height: 100vh; text-align: center;
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

  // --- Screen 1 — Welcome Screen ---
  const c = config.consent;
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:720px; text-align:left;">
        <h2 style="text-align:center;">Welcome to Our Study</h2>

        <h3>General Information</h3>
        <p>This experiment is part of a study that investigates cognitive processes related 
           to visual perception.<br>
           You will be shown sets of object images and asked to arrange them based on how 
           visually similar they appear to you. The task takes approximately
           <strong>${c.study_duration_minutes} minutes</strong>.</p>

        <h3>Participation</h3>
        <p>Your participation is voluntary. You may stop at any time by returning your
           Prolific submission; this will not affect your Prolific account in any way.
           You will receive Prolific payment for your time.</p>

        <h3>Data and Privacy</h3>
        <p>Your responses will be stored for scientific analysis, linked to a participant
           code. The link between the code and your identity is kept separately from the
           data. If results are published, they will refer to group-level statistics and
           will not identify you in any way.<br>
           Following the <i>Open Science</i> principle, anonymized data may be shared with 
           other researchers or deposited on public repositories. Any information that
           could identify you will be removed before data are shared or made public.</p>
      </div>`,
    choices: ['Continue to consent'],
  });

  // --- Consent: Screen 2 — Declaration of Consent ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:720px; text-align:left;">
        <h2 style="text-align:center;">Declaration of Consent</h2>
        <p>By clicking <strong>"I agree to participate"</strong> below, you confirm that:</p>
        <ul>
          <li>You are 18 years of age or older.</li>
          <li>You have read and understood the participant information on the previous page.</li>
          <li>You understand that your participation is voluntary and you may withdraw at
              any time without penalty by returning your submission on Prolific.</li>
          <li>You agree to participate in this study in exchange for Prolific payment.</li>
        </ul>
        
        <h3>Contact</h3>
        <p>For any questions, please contact:<br>
           Researcher: ${c.researcher_name} &mdash; <a href="mailto:${c.researcher_email}" style="color:#aad;">${c.researcher_email}</a><br>
           Principal Investigator: ${c.pi_name} &mdash; <a href="mailto:${c.pi_email}" style="color:#aad;">${c.pi_email}</a><br>
           ${c.lab_name}, ${c.institution}<br>
           Tel: ${c.lab_phone}</p>
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
        <p><strong>Important:</strong> Read the instruction at the top of every trial
           carefully — the task may vary from trial to trial.</p>
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
    prompt: '<p style="font-size:0.9em;"><strong>Your task:</strong> Arrange these images by their ' +
        '<strong>visual similarity.</strong><br>' +
        'PRACTICE — this trial will <strong>not</strong> be recorded.</p>',
    counter_text_unfinished:  '',
    counter_text_finished:    '',
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
    // NOTE: header_text must go in `prompt`, NOT counter_text_*.
    // counter_text_* wraps its content in a <p>; putting a <p> inside a <p> is
    // invalid HTML, and the browser auto-corrects it by splitting them.  When
    // stim_starts_inside:true the plugin then immediately overwrites the counter
    // innerHTML, rendering the text a second time → duplicate instructions.
    const header_text = trial.type === 'catch'
        ? '<p style="font-size:0.9em;"><strong>Your task:</strong> Place all images in the ' +
          '<strong>' + trial.target_location + '</strong> of the screen.</p>'
        : '<p style="font-size:0.9em;"><strong>Your task:</strong> Arrange the images by visual similarity. ' +
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
      prompt:                   header_text,
      counter_text_unfinished:  '',
      counter_text_finished:    '',
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
        data.participant_id        = PID;
        data.shine_variant         = assignedVariant;
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
          data.centroid_x             = centroid.x / sortW;
          data.centroid_y             = centroid.y / sortH;
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

        if (mode === 'debug') {
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
