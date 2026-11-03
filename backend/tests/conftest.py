import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="roto_test_")
os.environ["ROTOBRUSH_DATA"] = str(Path(_tmp) / "data")
os.environ.setdefault("ROTOBRUSH_NO_WARMUP", "1")  # no background model loading while tests run: deterministic and much lighter
os.environ["ROTOBRUSH_WORKSPACE"] = str(Path(_tmp) / "ws")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
