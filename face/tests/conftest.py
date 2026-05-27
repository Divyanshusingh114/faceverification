"""
Test-time defaults applied BEFORE any project module is imported.

`get_settings()` is cached by `@lru_cache`, so the env must be primed here
or test imports will pick up production defaults (e.g. /var/log paths).
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ENV", "dev")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault(
    "AUDIT_LOG_PATH",
    str(Path(tempfile.gettempdir()) / "aav_audit_test.log"),
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

# Make the project root importable regardless of where pytest is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional insightface stub.
# Real CI / Docker have insightface installed and we use it as-is. Local dev
# without python3-dev headers can't compile it; stub a no-op so the tests
# (which monkeypatch the model anyway) still run.
try:  # noqa: SIM105
    import insightface  # noqa: F401
except ImportError:
    import types

    _stub = types.ModuleType("insightface")
    _stub_app = types.ModuleType("insightface.app")

    class _StubFaceAnalysis:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def prepare(self, *args, **kwargs) -> None:
            pass

        def get(self, _img):  # noqa: ANN001
            return []

    _stub_app.FaceAnalysis = _StubFaceAnalysis
    sys.modules["insightface"] = _stub
    sys.modules["insightface.app"] = _stub_app
