# Trident-XLM: Turkish Explainability LLM

Trident-XLM is the explainability LLM layer for the TRIDENT-Net project, providing Turkish language summaries and reports based on telemetry data from multi-modal target detection systems.

## 🎯 Overview

This package provides:
- **Model-only NLP processing**: Consumes only telemetry facts, never operational placeholders
- **Dual model architecture**: Fast one-liners (Flan-T5) + detailed reports (mT5)
- **Turkish language generation**: Natural, technical prose for engineering audiences
- **Content guards**: Prevents operational terminology and placeholder injection
- **Multiple style presets**: `resmi`, `madde`, `anlatımcı` writing styles
- **CPU-optimized inference**: INT8 quantized models for edge deployment

## 🏗️ Architecture

```
Telemetry Data → Prompt Builder → LLM Models → Guards → Turkish Text
     ↓                ↓              ↓          ↓         ↓
- Probabilities   - Style-aware   - Flan-T5    - No ops  - One-liner
- Flags           - Turkish       - mT5        - No <>   - Report  
- Explanations    - Structured    - LoRA       - Style   - Safe
- Contributions   - Facts-only    - Quantized  - Check   - Content
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/YKesX/Trident-XLM.git
cd Trident-XLM

# Install dependencies (Torch-only recommended)
pip install -r requirements.txt

# If you hit TensorFlow/NumPy ABI issues on Windows, force torch-only paths:
# (CLI and scripts do this automatically)
# set TRANSFORMERS_NO_TF=1
# set TRANSFORMERS_NO_FLAX=1
# set USE_TORCH=1

# Test installation
python cli.py test
```

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# From telemetry JSONL file
python -m report_llm.build_dataset \
  --telemetry report_llm/data/telemetry.jsonl \
  --out_dir report_llm/data

# Outputs: train.jsonl, val.jsonl, test.jsonl
```

### 2. Train Models

```bash
# One-liner model (Flan-T5 + LoRA)
python -m report_llm.train_flan_one_liner \
  --train report_llm/data/train.jsonl \
  --val report_llm/data/val.jsonl \
  --out report_llm/exports/flan_one_liner

# Report model (mT5 + LoRA)
python -m report_llm.train_mt5_report \
  --train report_llm/data/train.jsonl \
  --val report_llm/data/val.jsonl \
  --out report_llm/exports/mt5_report
```

### 3. Quantize for CPU

```bash
# CPU-safe export (no GPU/bitsandbytes required)
python -m report_llm.quantize \
  --model_in report_llm/exports/flan_one_liner \
  --model_out report_llm/exports/flan_one_liner_int8

python -m report_llm.quantize \
  --model_in report_llm/exports/mt5_report \
  --model_out report_llm/exports/mt5_report_int8
```

### 4. Run Inference

```bash
# Using CLI
python cli.py inference \
  --telemetry report_llm/data/telemetry.jsonl \
  --model-sync report_llm/exports/flan_one_liner_int8 \
  --model-async report_llm/exports/mt5_report_int8 \
  --output outputs/trained_inference.json

# Using Python API
from report_llm import make_one_liner, make_report, build_inputs_for_llm

# Build prompt from telemetry
prompt = build_inputs_for_llm(telemetry_data, style="resmi")

# Generate outputs
one_liner = make_one_liner("path/to/flan_model", prompt)
report = make_report("path/to/mt5_model", prompt)
```

### 5. Randomized Demo (no dataset needed)

Quickly see Turkish outputs with synthetic telemetry:

```powershell
python .\random_demo.py
```

It will print a prompt, generate a one-liner and a report. If trained model folders are not present, it falls back to base HF models or a small Turkish template so you always get output.

## 📋 Data Format

### Input: TelemetryNLIn

```python
from report_llm.types import TelemetryNLIn, Contribution

telemetry = TelemetryNLIn(
    # Core probabilities (calibrated by pipeline)
    p_hit_calib=0.82, p_kill_calib=0.74,
    p_hit_masked=0.78, p_kill_masked=0.71,
    spoof_risk=0.15,
    
    # Quality flags
    flags={"mask_applied": True, "roi_coverage": 0.83},
    
    # Explainability summaries
    exp={"attn_hotspots": ["burun", "gövde"], "gradcam_roi": "merkez"},
    
    # Metadata safe for NL use
    meta={"sensor_mode": "RGB+IR", "speed_kph": 45},
    
    # Scored contributions
    contributions=[
        Contribution("RCS", "RADAR (Ka-band)", "pos", 39.40, "güçlü yansıma"),
        Contribution("Gürültü", "EO (LWIR)", "neg", -8.25, "atmosferik")
    ]
)
```

### Output: Turkish Text

```
One-liner: "Yüksek güvenilirlik seviyesinde hedef tespiti gerçekleştirildi."

Report: "Birincil Gerekçe: Radar kesit alanı ve Doppler frekans ölçümleri 
güçlü ve tutarlı sinyal karakteristikleri göstermiştir..."
```

## 🛡️ Content Guards

The system enforces strict content policies:

### Forbidden Content
- **Operational terms**: `ateş`, `ateşle`, `nişan`, `vur`, `angaje ol`
- **Angle brackets**: `<system>`, `<gun>`, `<operator>`, etc.
- **Tactical advice**: System provides only technical assessment

### Allowed Content
- **Technical terms**: `RCS`, `Doppler`, `LWIR`, `SpoofShield`
- **Assessment language**: Confidence levels, signal quality, factors
- **Turkish prose**: Natural engineering-focused text

## 🎨 Style Presets

| Style | Description | Use Case |
|-------|-------------|----------|
| `resmi` | Formal, official tone | Reports, documentation |
| `madde` | Bullet-point style | Quick summaries |
| `anlatımcı` | Narrative, explanatory | Training materials |

## 🔧 CLI Usage

```bash
# Generate prompt from telemetry
python cli.py prompt --telemetry data.jsonl --style resmi

# Train models with quantization
python cli.py train --train train.jsonl --val val.jsonl --quantize --epochs-flan 3 --epochs-mt5 2 --max-steps 0

# Run inference
python cli.py inference --telemetry sample.jsonl --output results.json

# Run tests
python cli.py test

## 📚 Examples

Additional demonstration and legacy scripts are moved into `examples/` to keep the root clean. Prefer `cli.py` and `random_demo.py` for day-to-day use. See `examples/README.md` for a list.
```

## 🧪 Testing

```bash
# Run all tests
python cli.py test

# Core functionality (no ML dependencies)
python test_core.py

# Logic validation 
python test_logic.py

# Comprehensive unit tests
python report_llm/tests/test_comprehensive.py
```

## 📁 Project Structure

```
Trident-XLM/
├── report_llm/              # Core package
│   ├── types.py             # Data structures
│   ├── prompt_builder.py    # Prompt generation
│   ├── build_dataset.py     # Training data creation
│   ├── train_flan_one_liner.py  # One-liner training
│   ├── train_mt5_report.py  # Report training
│   ├── summarizer_sync.py   # Fast inference
│   ├── summarizer_async.py  # Async inference
│   ├── quantize.py          # Model quantization
│   ├── style_guard.py       # Content validation
│   ├── data/                # Training data
│   ├── exports/             # Trained models
│   └── tests/               # Unit tests
├── cli.py                   # Command-line interface
├── test_core.py             # Core functionality test
├── test_logic.py            # Logic validation test
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔄 Integration with TRIDENT-Net

For integration into the main TRIDENT-Net pipeline:

1. **Replace stub reporter** in `trident/xai_text/small_llm_reporter.py`
2. **Enable in config**: Set `SmallLLMReporter.enabled: true` in `tasks.yml`  
3. **Wire outputs**: Add `report` field to output schema
4. **Load models**: Use quantized models for CPU inference

```python
# Example integration
from report_llm import make_one_liner, build_inputs_for_llm

def generate_report(telemetry_data):
    prompt = build_inputs_for_llm(telemetry_data, style="resmi")
    one_liner = make_one_liner(MODEL_SYNC_PATH, prompt)
    # Queue async report generation...
    return one_liner
```

## 🚫 Limitations

- **Model-only scope**: Does not handle `<angle>` placeholders or timing injection
- **Turkish language**: Optimized for Turkish technical writing only
- **Telemetry-dependent**: Requires structured telemetry input format
- **No operational output**: Intentionally avoids tactical/operational language

## 📊 Performance

- **One-liner**: short output (≤32 tokens)
- **Report**: medium output (≤192 tokens)
- **Memory**: depends on model size; quantized copies save RAM
- **Accuracy**: Depends on training data quality and domain match

## 🤝 Contributing

1. Follow existing code structure and patterns
2. Add tests for new functionality
3. Ensure Turkish language output quality
4. Validate guard policies are enforced
5. Test with representative telemetry data

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues related to:
- **Training/Inference**: Check model paths and data format
- **Guards failing**: Verify output contains no forbidden content  
- **Integration**: Ensure telemetry data matches expected schema
- **Performance**: Consider using quantized models and CPU optimization

---

**Trident-XLM** - Turkish explainability LLM for multi-modal target detection systems.