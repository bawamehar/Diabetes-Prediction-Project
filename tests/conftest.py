import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# src/ holds flat modules (not a package); make them importable from the tests.
sys.path.insert(0, str(ROOT / "src"))
# The optional api/ package is imported as `api.fastapi`.
sys.path.insert(0, str(ROOT))
