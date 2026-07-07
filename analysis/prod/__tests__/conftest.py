import sys
from pathlib import Path

# Make the repo root importable so 'from analysis.prod.parser import ...' works
# regardless of the working directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# NOTE: keep this file to just the sys.path bootstrap. Test-directories with no
# __init__.py are collected as top-level modules, so a second CSV-writing
# helper module literally named "conftest" (e.g. analysis/utils/__tests__/conftest.py)
# would collide with this one the moment both get collected in the same pytest
# invocation (whichever loads first "wins" for every `from conftest import ...`
# elsewhere) -- see prod_csv_helpers.py, which is deliberately NOT named
# conftest.py so it can't collide with anything.
