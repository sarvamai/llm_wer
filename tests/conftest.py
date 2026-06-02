"""Stub heavy optional deps so core logic tests run without ML packages."""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "indicnlp", "indicnlp.normalize", "indicnlp.normalize.indic_normalize",
    "transformers",
    "joblib",
    "gspread", "gspread.exceptions",
    "google", "google.auth", "google.oauth2", "google.oauth2.service_account",
    "google.cloud", "google.cloud.aiplatform",
    "vertexai", "vertexai.generative_models",
    # stub the whole llm_api module — tests don't use ChatCompletionsAPI
    "llm_api",
]
for _m in _STUBS:
    if _m not in sys.modules:
        mock = MagicMock()
        mock.__path__ = []
        sys.modules[_m] = mock

# IndicNormalizerFactory must be importable as a class
import sys
sys.modules["indicnlp.normalize.indic_normalize"].IndicNormalizerFactory = MagicMock
