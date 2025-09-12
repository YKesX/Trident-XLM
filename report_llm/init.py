# TRIDENT report_llm package
# This package handles ONLY the model-side NLP: building prompts from telemetry,
# training small seq2seq models (flan-t5-small for one-liners, mt5-small for long reports),
# and generating text sections "from scratch" using ONLY telemetry inputs.
# No wrapper-side placeholders (<system>, <gun>, etc.) are managed here.

# This package contains fine-tuned Turkish LLM models for explainability in TRIDENT,
# which take in filtered telemetry data (no angle-bracket wrappers) and produce
# Turkish text summaries in different styles without operational words.
from .types import TelemetryNLIn, Contribution, StylePreset
from .prompt_builder import build_inputs_for_llm
from .summarizer_sync import make_one_liner
from .summarizer_async import make_report, make_report_async
