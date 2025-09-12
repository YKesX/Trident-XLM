# TRIDENT report_llm package
# This package handles ONLY the model-side NLP: building prompts from telemetry,
# training small seq2seq models (flan-t5-small for one-liners, mt5-small for long reports),
# and generating text sections "from scratch" using ONLY telemetry inputs.
# No wrapper-side placeholders (<system>, <gun>, etc.) are managed here.
