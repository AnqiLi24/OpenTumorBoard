"""Entry point for `python src/slides` (or `python -m slides` from src/)."""
import sys
import os

# When run as `python src/slides`, Python sets sys.path[0] to the package
# directory itself. We need its parent (src/) so that:
#   1. `slides` is importable as a proper package (for relative imports).
#   2. `gpt_client` (a sibling of the package in src/) is importable.
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from slides._cli import main  # noqa: E402

main()
