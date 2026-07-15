"""Root conftest so `import src...` and `import backend...` work under pytest
regardless of invocation directory (no package __init__.py files in this repo
yet, so pytest's own rootdir-insertion doesn't add the repo root by itself).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
