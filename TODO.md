# TODO

- **Unify `analysis/pilot/` and `analysis/prod/` data loading + CLI.**
  `analysis/prod/parser.py` (`load_prod_data`) is a near-duplicate of
  `analysis/utils/parser.py` (`load_pilot_data`) -- same demographics/session
  CSV schema, same trial parsing -- differing only in the demographics
  filename glob and prod's extra "session must have all 20 trials + 4 catch
  trials" completeness check. Likewise `analysis/prod/show_trials.py` is a
  near-duplicate of `analysis/pilot/show_trials.py` (same --version/
  --participant/--list CLI, just pointed at a different loader + data dir).
  Once prod data collection settles, merge both loaders into one function
  (data_dir + demographics-glob + completeness-predicate as parameters) and
  merge the two show_trials.py scripts into a single script parameterized by
  dataset (pilot vs prod).
