"""Root conftest -- makes the repo importable no matter how pytest is invoked.

`services/` lives at the repo root rather than under `src/`, so the editable
install does not put it on the import path. Depending on how pytest is started
(`pytest` vs `python -m pytest`, from the repo root or elsewhere) that package
may or may not be importable, which made the entire API suite fail with
`ModuleNotFoundError: No module named 'services'` on a clean checkout.

pytest imports the rootdir conftest before collecting anything, on every
version and every invocation style, so doing the insert here is the one
approach that cannot be bypassed. The `pythonpath` setting in pyproject.toml
covers the same ground for pytest >= 7; this is the belt to that pair of
braces, and it costs three lines.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
