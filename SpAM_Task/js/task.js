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
      '@keyframes spam-spin { to { transform: rotate(360deg); } }',
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
  // 5. Responsive layout
  //    Initialised now (pre-fullscreen) so downstream code can reference the
  //    variables; recomputed in the fullscreen trial's on_finish using the
  //    actual fullscreen viewport dimensions.
  // ---------------------------------------------------------------------------
  let { sortW, sortH, stimSize } = computeLayout(
    window.innerWidth, window.innerHeight, config,
  );
  if (mode === 'debug') console.log('[SpAM] Layout (pre-fullscreen):', { sortW, sortH, stimSize });

  // jsPsychFreeSort only recognises 'square' (rectangular) and 'ellipse'.
  // Our config uses 'rect'/'ellipse'; map here so the inside-check uses the
  // correct geometry (passing an unrecognised value falls through to ellipse,
  // which would exclude images placed in the corners of a rectangular arena).
  const pluginShape = config.display.sort_area_shape === 'ellipse' ? 'ellipse' : 'square';

  // ---------------------------------------------------------------------------
  // Catch-trial compliance blocking
  //   Polls image positions every 250 ms.  While the cluster centroid is not
  //   near targetLocation the "Done" button is disabled and a warning is shown.
  //   Called from on_load for both the example and the real catch trials.
  // ---------------------------------------------------------------------------
  // jsPsych 7 calls on_load with no arguments; the display element must be
  // retrieved via jsPsych.getDisplayElement().
  function attachTrialTimer(minMs) {
    const displayEl = jsPsych.getDisplayElement();
    const btn = displayEl.querySelector('#jspsych-free-sort-done-btn');
    if (!btn) return;

    const notice = document.createElement('p');
    notice.id = 'trial-timer-notice';
    notice.style.cssText = 'font-size:0.85em; margin:4px 0 0; min-height:1.2em;';
    btn.insertAdjacentElement('afterend', notice);

    btn.disabled = true;
    const startTime = Date.now();

    function tick() {
      const remaining = Math.ceil((minMs - (Date.now() - startTime)) / 1000);
      if (remaining > 0) {
        notice.textContent = 'Please keep arranging — you can continue in ' + remaining + 's.';
      } else {
        btn.disabled = false;
        notice.textContent = 'Press the button when you are done arranging the images.';
        clearInterval(timer);
      }
    }

    const timer = setInterval(tick, 1000);
    tick();
  }

  function attachCatchCompliance(targetLocation) {
    const displayEl = jsPsych.getDisplayElement();
    const btn = displayEl.querySelector('#jspsych-free-sort-done-btn');
    if (!btn) return () => {};

    // Warning element injected immediately below the button.
    const warning = document.createElement('p');
    warning.id = 'catch-compliance-warning';
    warning.style.cssText = 'color:#FFD600; font-size:0.85em; margin:4px 0 0; min-height:1.2em;';
    btn.insertAdjacentElement('afterend', warning);

    function check() {
      const items = displayEl.querySelectorAll('.jspsych-free-sort-draggable');
      if (!items.length) return;
      const locations = Array.from(items).map(el => ({
        src: el.dataset.src,
        x:   parseInt(el.style.left, 10),
        y:   parseInt(el.style.top,  10),
      }));
      const compliant = allImagesNearTarget(
        locations, targetLocation, sortW, sortH,
        config.catch_trials.location_tolerance,
      );
      btn.disabled        = !compliant;
      warning.textContent = compliant
        ? ''
        : 'Please follow the trial instructions before continuing.';
    }

    const timer = setInterval(check, 250);
    check(); // immediate check so the button is blocked from the first visible frame
    return () => clearInterval(timer);
  }

  // ---------------------------------------------------------------------------
  // 6. Trial lists
  //    buildTrialLists expects rng as a function (not a seed integer).
  //    rng must not be called between here and buildTrialLists — any prior call
  //    shifts the sequence and breaks per-PID reproducibility.
  //    insertCatchTrials uses the same rng instance (continued sequence) so
  //    catch location assignment is also seeded and recorded in trial data.
  // ---------------------------------------------------------------------------
  const distinctTrials = buildTrialLists(imageUrls, config, rng);
  const mainTrials = insertTrialRepeats(distinctTrials, config, rng);
  const allTrials   = insertCatchTrials(mainTrials, catchUrls, config, rng);

  // Maps each main trial's trialId to its final position in allTrials, so a
  // repeat's repeatOfTrialId can be resolved to the original's saved trial_index.
  const trialIdToIndex = {};
  allTrials.forEach((trial, idx) => {
    if (trial.type === 'main') trialIdToIndex[trial.trialId] = idx;
  });

  if (mode === 'debug') {
    console.log('[SpAM] Trial sequence:', allTrials.map((t, i) => i + ':' + t.type + (t.isRepeat ? ' (repeat)' : '')));
  }

  // ---------------------------------------------------------------------------
  // 7. jsPsych initialisation
  // ---------------------------------------------------------------------------
  const jsPsych = initJsPsych({
    auto_preload: false, // we preload per-trial via jsPsychPreload nodes

    // Local only: download a filtered CSV and show the end screen.
    // On Pavlovia, the redirect is handled in the jsPsychPavlovia finish
    // trial's on_finish (below), which fires only after data is uploaded.
    on_finish: function() {
      if (isPavlovia) return;
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
      document.body.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:100vh;text-align:center;">
          <div>
            <p>Press <strong>Esc</strong> to exit full screen.</p>
            <p>You may now close this window.</p>
          </div>
        </div>`;
    },
  });

  // Session-level fields written to every trial row automatically.
  // sort_area_width/height give the rectangle boundaries (top-left is always
  // 0,0 in final_locations coords; bottom-right is sortW, sortH).
  jsPsych.data.addProperties({
    participant_id:   PID,
    task_version:     config.deployment.version,
    deployment_mode:  mode,
    shine_variant:    assignedVariant,
    sort_area_width:  sortW,
    sort_area_height: sortH,
  });

  // ---------------------------------------------------------------------------
  // 9. Timeline
  // ---------------------------------------------------------------------------
  const timeline = [];

  // --- Pavlovia init (Pavlovia only) ---
  if (isPavlovia) {
    timeline.push({ type: jsPsychPavlovia, command: 'init' });
  }

  // --- init consent form ---
  const c = config.consent;

  // --- Screen 1 — General Info ---
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:720px; text-align:left;">
        <h2 style="text-align:center;">Welcome to Our Study</h2>

        <h3>General Information</h3>        
        <p>This experiment is part of a study that investigates cognitive processes related 
           to visual perception.<br>
           You will be shown sets of object images and asked to arrange them based on how 
           visually similar they appear to you.<br>
           The task takes approximately <strong>${c.study_duration_minutes} minutes</strong>. 
        </p>

        <h3>Participation and Compensation</h3>
        <p>Your participation is voluntary. You may stop at any time by returning to the
           Prolific dashboard and clicking <strong>"Stop without completing"</strong> &ndash;
           this will not affect your Prolific account in any way.<br>
           You will receive payment from Prolific for your time. Otherwise, the experiment
           will not benefit you personally, but we expect the results to improve our
           understanding of human cognitive and neural function.
        </p>
      </div>`,
    choices: ['Continue'],
  });

  // --- Screen 2 — Privacy & Contact Info ---
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:720px; text-align:left;">

        <h3>Data and Privacy</h3>
        <p>Your responses will be stored for scientific analysis and research purposes,
           linked to a participant code that identifies you. The link between the code 
           and your personal information is kept separately from the recorded data. 
           If results are published, they will typically refer to group-level statistics 
           and will not identify you in any way.<br>
           Following the <i>Open Science</i> principle, we may share the collected data 
           with other researchers or post them on public repositories for the sake of future 
           scientific analysis and scrutiny. Any personal information that could identify 
           you will be removed or changed before files are shared with other researchers or 
           results are made public.
        </p>
        
        <h3>Contact</h3>
        <p>For any questions, please contact:<br>
           Researcher: <a href="mailto:${c.researcher_email}" style="color:#0000FF;">${c.researcher_name}</a><br>
           Principal Investigator: <a href="mailto:${c.pi_email}" style="color:#0000FF;">Prof. ${c.pi_name}</a><br>
           ${c.lab_name}, ${c.institution}<br>
           Tel: ${c.lab_phone}
        </p>
      </div>`,
    choices: ['Continue'],
  });

  // --- Screen 3 — Declaration of Consent ---
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
              any time without penalty by returning to the Prolific dashboard and clicking
              <strong>"Stop without completing"</strong>.</li>
          <li>You agree to participate in this study in exchange for Prolific payment.</li>
        </ul>
      </div>`,
    choices: ['I agree to participate', 'I do not wish to participate'],
    on_finish: function(data) {
      if (data.response === 1) {
        // Participant declined consent — redirect to Prolific no-consent URL if set,
        // otherwise show a neutral end screen.
        if (config.deployment.prolific_no_consent_url) {
          window.location.href = config.deployment.prolific_no_consent_url;
        } else {
          jsPsych.abortExperiment(
            '<p style="text-align:center;margin-top:20%;">You have chosen not to participate.<br>You may now close this window.</p>'
          );
        }
      }
    },
  });

  // --- Enter fullscreen ---
  // on_finish recomputes layout using the actual fullscreen viewport so that
  // sort-area dimensions reflect the full screen, not the windowed browser.
  timeline.push({
    type:            jsPsychFullscreen,
    fullscreen_mode: true,
    message:         '<p>The task will now switch to fullscreen mode.</p>',
    button_label:    'Continue',
    on_finish: function() {
      ({ sortW, sortH, stimSize } = computeLayout(
        window.innerWidth, window.innerHeight, config,
      ));
      // Overwrite the sort_area dimensions logged to every trial row now that
      // the true fullscreen viewport is known.
      jsPsych.data.addProperties({ sort_area_width: sortW, sort_area_height: sortH });
      if (mode === 'debug') console.log('[SpAM] Layout (post-fullscreen):', { sortW, sortH, stimSize });
    },
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
        <p>There are no right or wrong answers &ndash; go with your first impression.</p>
        <p>You will first do two short <strong>practice trials</strong>,
           then ${allTrials.length} real trials.</p>
        <p>Do not use the back button.</p>
        <p><strong>Important:</strong> Read the instruction at the top of every trial
           carefully &ndash; the task may change occasionally.</p>
      </div>`,
    choices: ['Continue'],
  });

  // --- Before/After Examples ---
  // Two side-by-side before→after screenshot pairs illustrating what a
  // completed trial looks like.
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:960px; text-align:center;">
        <h2>Example</h2>
        <p>Below are two examples of completed trials.<br>
           Images that <strong>look similar</strong> are placed <strong>close together</strong>.<br>
           images that <strong>look different</strong> are placed <strong>far apart</strong>.</p>
        <div style="display:flex; flex-direction:column; gap:20px; margin-top:20px;">
          <div style="display:flex; align-items:center; justify-content:center; gap:16px;">
            <img src="SpAM_Task/assets/examples/before1.png"
                 style="max-width:420px; max-height:240px; border:2px solid #555;">
            <span style="font-size:2em; flex-shrink:0;">&rarr;</span>
            <img src="SpAM_Task/assets/examples/after1.png"
                 style="max-width:420px; max-height:240px; border:2px solid #555;">
          </div>
          <div style="display:flex; align-items:center; justify-content:center; gap:16px;">
            <img src="SpAM_Task/assets/examples/before2.png"
                 style="max-width:420px; max-height:240px; border:2px solid #555;">
            <span style="font-size:2em; flex-shrink:0;">&rarr;</span>
            <img src="SpAM_Task/assets/examples/after2.png"
                 style="max-width:420px; max-height:240px; border:2px solid #555;">
          </div>
        </div>
      </div>`,
    choices: ['Start practice'],
  });

  // --- Practice Experimental Trial ---
  // Shows participants what an experimental trial looks like before the real trials start.
  timeline.push({
    type:            jsPsychFreeSort,
    stimuli:         practiceUrls.slice(0, config.catch_trials.images_per_trial),
    sort_area_width:  sortW,
    sort_area_height: sortH,
    stim_width:        stimSize,
    stim_height:       stimSize,
    sort_area_shape:      pluginShape,
    stim_starts_inside:   config.display.stim_starts_inside,
    column_spread_factor: config.display.column_spread_factor,
    prompt: '<p style="font-size:0.9em;"><strong>Your task:</strong> Arrange these images by their ' +
        '<strong>visual similarity.</strong><br><br>' +
        'PRACTICE &ndash; this trial will <strong>not</strong> be recorded.</p>',
    counter_text_unfinished:  '',
    counter_text_finished:    '',
    on_finish: function(data) {
      data.trial_type = 'practice';
    },
  });

  // --- Pre Catch-Practice Transition ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: '<p>Very Good!</p>' +
              '<p>Most trials in the experiment will be similar to the one you have just seen.<br>' +
              'However, occasionally you will see trials with unique instructions.<br>' +
              'Read the instructions carefully, and perform the task according to them.</p>' +
              '<p>Click on the button to see an example of a unique trial.</p>',
    choices:  ['Continue'],
  });

  // --- Practice Catch Trial ---
  // Shows participants what a catch trial looks like before the real trials start.
  // Always directs to "bottom right corner".
  timeline.push({
    type:             jsPsychFreeSort,
    stimuli:          catchUrls.slice(0, config.catch_trials.images_per_trial),
    sort_area_width:  sortW,
    sort_area_height: sortH,
    stim_width:        stimSize,
    stim_height:       stimSize,
    sort_area_shape:      pluginShape,
    stim_starts_inside:   config.display.stim_starts_inside,
    column_spread_factor: config.display.column_spread_factor,
    prompt: '<p style="font-size:0.9em;"><strong>Your task:</strong> Place all images in the ' +
            '<strong>bottom right corner</strong> of the screen.<br><br>' +
            'PRACTICE &ndash; this trial will <strong>not</strong> be recorded.</p>',
    counter_text_unfinished:  '',
    counter_text_finished:    '',
    on_load() {
      attachCatchCompliance('bottom right corner');
    },
    on_finish: function(data) {
      data.trial_type = 'practice_catch';
    },
  });

  // --- Post-practice transition ---
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: '<p>Practice complete. The real experiment begins now.</p>' +
              '<p>Remember: close together = <strong>visually similar</strong>, far apart = <strong>visually different</strong>.</p>',
    choices:  ['Continue'],
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
        : '<p style="font-size:0.9em;"><strong>Your task:</strong> Arrange the images by <strong>visual</strong> similarity. ' +
          'Close together = <strong>visually similar</strong> &nbsp;|&nbsp; Far apart = <strong>visually different</strong>.</p>';
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
      on_load: trial.type === 'catch'
        ? function () { attachCatchCompliance(trial.target_location); }
        : function () { attachTrialTimer(mode === 'debug' ? 5000 : config.quality_control.min_trial_rt_ms); },
      on_finish: function(data) {
        // QC metrics
        const pairs = computePairwiseDistances(
          data.final_locations,
          sortW,
          sortH,
        );
        const distances = pairs.map(p => p.distance);
        const sd = computeSD(distances);
        const numMoves = data.moves.length;

        data.num_moves = numMoves;

        // trial_type: 'trial_N' for main trials, 'catch_N' for catch trials
        // (1-based running index within each category)
        data.trial_type  = trial.type === 'catch'
          ? 'catch_' + (++catchCount)
          : 'trial_' + (++mainCount);
        data.trial_index = idx;

        if (trial.type === 'main') {
          data.is_trial_repeat     = !!trial.isRepeat;
          data.repeat_of_trial_index = trial.isRepeat ? trialIdToIndex[trial.repeatOfTrialId] : null;
        }

        if (trial.type === 'catch') {
          // Catch-trial QC: cluster tightness + centroid proximity to target.
          // computeCentroid and isCentroidNearTarget added with new catch design.
          const centroid   = computeCentroid(data.final_locations);
          const clusterMean = distances.reduce((a, b) => a + b, 0) / (distances.length || 1);
          const locationOk = allImagesNearTarget(
            data.final_locations, trial.target_location,
            sortW, sortH,
            config.catch_trials.location_tolerance,
          );
          data.catch_trial_target_location = trial.target_location;
          data.centroid_x             = centroid.x / sortW;
          data.centroid_y             = centroid.y / sortH;
          data.cluster_mean_distance  = clusterMean;
          data.qc_flag = computeCatchQcFlag(clusterMean, sd, locationOk, numMoves, trial.images.length, config);
        } else {
          data.qc_flag = computeMainQcFlag(sd, numMoves, trial.images.length, config);
        }

        data.pairwise_distances = JSON.stringify(pairs);

        if (mode === 'debug') {
          console.log(
            '[SpAM] Trial', idx, '(' + trial.type + ')',
            '| sd=' + sd.toFixed(4),
            '| moves=' + numMoves + '/' + trial.images.length,
            '| rt=' + data.rt + 'ms',
            '| qc_flag=' + data.qc_flag,
          );
        }
      },
    });
  });

  // --- Pavlovia finish (Pavlovia only) ---
  // Must come BEFORE the debrief so that data is fully saved (upload + session
  // close) before the participant can click the Prolific redirect button.
  // on_load injects a spinner while the plugin's async API calls run; the trial
  // ends automatically when the plugin calls end_trial() — no timeout needed.
  if (isPavlovia) {
    timeline.push({
      type: jsPsychPavlovia,
      command: 'finish',
      on_load: function() {
        jsPsych.getDisplayElement().innerHTML = `
          <div style="display:flex;align-items:center;justify-content:center;height:100vh;">
            <div style="text-align:center;">
              <div style="
                width: 48px; height: 48px;
                border: 5px solid rgba(17,17,17,0.2);
                border-top-color: #111111;
                border-radius: 50%;
                animation: spam-spin 0.8s linear infinite;
                margin: 0 auto 24px;
              "></div>
              <p>Saving your responses&hellip;</p>
              <p style="font-size:0.85em; opacity:0.6;">Please do not close this window.</p>
            </div>
          </div>`;
      },
    });
  }

  // --- Debrief ---
  // On Pavlovia: shown after save completes; "Finish" redirects immediately.
  // Locally: shown last; global on_finish downloads the CSV.
  timeline.push({
    type:     jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="max-width:600px; text-align:center;">
        <h2>Thank you!</h2>
        <p>You have completed all trials and your responses have been saved.</p>
        <p>Click <strong>Finish</strong> to return to Prolific and receive your completion credit.</p>
      </div>`,
    choices: ['Finish'],
    on_finish: function() {
      const completionUrl = config.deployment.prolific_completion_url;
      if (isPavlovia && completionUrl) {
        window.location.href = completionUrl;
      }
    },
  });

  // ---------------------------------------------------------------------------
  // 10. Run
  // ---------------------------------------------------------------------------
  jsPsych.run(timeline);

}); // end DOMContentLoaded
