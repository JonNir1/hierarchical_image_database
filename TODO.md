# TODO

- **Unify `analysis/pilot/show_trials.py` and `analysis/prod/show_trials.py`.**
  Data loading is already unified: `analysis/utils/parser.py`'s
  `load_pilot_data` works for both `data/pilot` and `data/prod` (tries both
  demographics filename patterns, and picks the most-complete session file
  when a participant has more than one). The two `show_trials.py` scripts
  are otherwise identical -- same --version/--participant/--list CLI, just
  pointed at a different `data_dir`. Merge them into a single script
  parameterized by dataset (pilot vs prod), e.g. a `--dataset pilot|prod`
  flag or a positional data_dir argument.
