# Trident-XLM: Türkçe Açıklanabilirlik LLM'i

Trident-XLM, TRIDENT-Net projesinin açıklanabilirlik LLM katmanıdır ve çok modlu hedef tespit sistemlerinden gelen telemetri verilerine dayalı olarak Türkçe özetler ve raporlar sağlar.

## 🎯 Genel Bakış

Bu paket şunları sağlar:
- **Yalnızca model tabanlı NLP işleme**: Sadece telemetri verilerini kullanır, operasyonel yer tutucuları asla
- **İkili model mimarisi**: Hızlı tek satır özetler (Flan-T5) + detaylı raporlar (mT5)
- **Türkçe dil üretimi**: Mühendislik hedef kitlesi için doğal, teknik düzyazı
- **İçerik korumaları**: Operasyonel terminoloji ve yer tutucu enjeksiyonunu önler
- **Çoklu stil ön ayarları**: `resmi`, `madde`, `anlatımcı` yazım stilleri
- **CPU-optimized çıkarım**: Edge dağıtım için INT8 kuantize modeller

## 🏗️ Mimari

```
Telemetri Verisi → Prompt Oluşturucu → LLM Modelleri → Korumalar → Türkçe Metin
     ↓                ↓                    ↓          ↓           ↓
- Olasılıklar     - Stil-bilinçli      - Flan-T5    - Ops yok   - Tek satır
- Bayraklar       - Türkçe             - mT5        - <> yok    - Rapor  
- Açıklamalar     - Yapılandırılmış    - LoRA       - Stil      - Güvenli
- Katkılar        - Sadece gerçekler   - Kuantize   - Kontrol   - İçerik
```

## 📦 Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/YKesX/Trident-XLM.git
cd Trident-XLM

# Bağımlılıkları kurun (Yalnızca Torch önerilen)
pip install -r requirements.txt

# Windows'ta TensorFlow/NumPy ABI sorunları yaşıyorsanız, torch-only yollarını zorlayın:
# (CLI ve scriptler bunu otomatik yapar)
# set TRANSFORMERS_NO_TF=1
# set TRANSFORMERS_NO_FLAX=1
# set USE_TORCH=1

# Kurulumu test edin
python cli.py test
```

## 🚀 Hızlı Başlangıç

### 1. Eğitim Verisi Oluşturma

```bash
# Telemetri JSONL dosyasından
python -m report_llm.build_dataset \
  --telemetry report_llm/data/telemetry.jsonl \
  --out_dir report_llm/data

# Çıktılar: train.jsonl, val.jsonl, test.jsonl
```

### 2. Modelleri Eğitme

```bash
# Tek satır modeli (Flan-T5 + LoRA)
python -m report_llm.train_flan_one_liner \
  --train report_llm/data/train.jsonl \
  --val report_llm/data/val.jsonl \
  --out report_llm/exports/flan_one_liner

# Rapor modeli (mT5 + LoRA)
python -m report_llm.train_mt5_report \
  --train report_llm/data/train.jsonl \
  --val report_llm/data/val.jsonl \
  --out report_llm/exports/mt5_report
```

### 3. CPU için Kuantizasyon

```bash
# CPU-güvenli export (GPU/bitsandbytes gerekli değil)
python -m report_llm.quantize \
  --model_in report_llm/exports/flan_one_liner \
  --model_out report_llm/exports/flan_one_liner_int8

python -m report_llm.quantize \
  --model_in report_llm/exports/mt5_report \
  --model_out report_llm/exports/mt5_report_int8
```

### 4. Çıkarım Çalıştırma

```bash
# CLI kullanarak
python cli.py inference \
  --telemetry report_llm/data/telemetry.jsonl \
  --model-sync report_llm/exports/flan_one_liner_int8 \
  --model-async report_llm/exports/mt5_report_int8 \
  --output outputs/trained_inference.json

# Python API kullanarak
from report_llm import make_one_liner, make_report, build_inputs_for_llm

# Telemetriden prompt oluştur
prompt = build_inputs_for_llm(telemetry_data, style="resmi")

# Çıktıları üret
one_liner = make_one_liner("path/to/flan_model", prompt)
report = make_report("path/to/mt5_model", prompt)
```

### 5. Rastgele Demo (veri seti gerekli değil)

Sentetik telemetri ile Türkçe çıktıları hızlıca görün:

```powershell
python .\random_demo.py
```

Bir prompt yazdırır, tek satır ve rapor üretir. Eğitilmiş model klasörleri mevcut değilse, temel HF modellerine veya küçük Türkçe şablona geri döner, böylece her zaman çıktı alırsınız.

## 📋 Veri Formatı

### Giriş: TelemetryNLIn

```python
from report_llm.types import TelemetryNLIn, Contribution

telemetry = TelemetryNLIn(
    # Çekirdek olasılıklar (pipeline tarafından kalibre edilmiş)
    p_hit_calib=0.82, p_kill_calib=0.74,
    p_hit_masked=0.78, p_kill_masked=0.71,
    spoof_risk=0.15,
    
    # Kalite bayrakları
    flags={"mask_applied": True, "roi_coverage": 0.83},
    
    # Açıklanabilirlik özetleri
    exp={"attn_hotspots": ["burun", "gövde"], "gradcam_roi": "merkez"},
    
    # NL kullanımı için güvenli metadata
    meta={"sensor_mode": "RGB+IR", "speed_kph": 45},
    
    # Puanlanmış katkılar
    contributions=[
        Contribution("RCS", "RADAR (Ka-band)", "pos", 39.40, "güçlü yansıma"),
        Contribution("Gürültü", "EO (LWIR)", "neg", -8.25, "atmosferik")
    ]
)
```

### Çıkış: Türkçe Metin

```
Tek satır: "Yüksek güvenilirlik seviyesinde hedef tespiti gerçekleştirildi."

Rapor: "Birincil Gerekçe: Radar kesit alanı ve Doppler frekans ölçümleri 
güçlü ve tutarlı sinyal karakteristikleri göstermiştir..."
```

## 🛡️ İçerik Korumaları

Sistem sıkı içerik politikaları uygular:

### Yasak İçerik
- **Operasyonel terimler**: `ateş`, `ateşle`, `nişan`, `vur`, `angaje ol`
- **Açı parantezleri**: `<system>`, `<gun>`, `<operator>`, vb.
- **Taktiksel tavsiye**: Sistem yalnızca teknik değerlendirme sağlar

### İzin Verilen İçerik
- **Teknik terimler**: `RCS`, `Doppler`, `LWIR`, `SpoofShield`
- **Değerlendirme dili**: Güven seviyeleri, sinyal kalitesi, faktörler
- **Türkçe düzyazı**: Doğal mühendislik odaklı metin

## 🎨 Stil Ön Ayarları

| Stil | Açıklama | Kullanım Alanı |
|-------|----------|---------|
| `resmi` | Resmi, resmî ton | Raporlar, dokümantasyon |
| `madde` | Madde işareti stili | Hızlı özetler |
| `anlatımcı` | Anlatısal, açıklayıcı | Eğitim materyalleri |

## 🔧 CLI Kullanımı

```bash
# Telemetriden prompt üret
python cli.py prompt --telemetry data.jsonl --style resmi

# Kuantizasyonla modelleri eğit
python cli.py train --train train.jsonl --val val.jsonl --quantize --epochs-flan 3 --epochs-mt5 2 --max-steps 0

# Çıkarım çalıştır
python cli.py inference --telemetry sample.jsonl --output results.json

# Testleri çalıştır
python cli.py test

## 📚 Örnekler

Ek gösterim ve eski scriptler, kökü temiz tutmak için `examples/` klasörüne taşınmıştır. Günlük kullanım için `cli.py` ve `random_demo.py`'yi tercih edin. Liste için `examples/README.md`'ye bakın.
```

## 🧪 Test Etme

```bash
# Tüm testleri çalıştır
python cli.py test

# Çekirdek işlevsellik (ML bağımlılıkları yok)
python test_core.py

# Mantık doğrulama 
python test_logic.py

# Kapsamlı birim testleri
python report_llm/tests/test_comprehensive.py
```

## 📁 Proje Yapısı

```
Trident-XLM/
├── report_llm/              # Çekirdek paket
│   ├── types.py             # Veri yapıları
│   ├── prompt_builder.py    # Prompt üretimi
│   ├── build_dataset.py     # Eğitim verisi oluşturma
│   ├── train_flan_one_liner.py  # Tek satır eğitimi
│   ├── train_mt5_report.py  # Rapor eğitimi
│   ├── summarizer_sync.py   # Hızlı çıkarım
│   ├── summarizer_async.py  # Asenkron çıkarım
│   ├── quantize.py          # Model kuantizasyonu
│   ├── style_guard.py       # İçerik doğrulama
│   ├── data/                # Eğitim verisi
│   ├── exports/             # Eğitilmiş modeller
│   └── tests/               # Birim testleri
├── cli.py                   # Komut satırı arayüzü
├── test_core.py             # Çekirdek işlevsellik testi
├── test_logic.py            # Mantık doğrulama testi
├── requirements.txt         # Bağımlılıklar
└── README.md               # Bu dosya
```

## 🔄 TRIDENT-Net ile Entegrasyon

Ana TRIDENT-Net pipeline'ına entegrasyon için:

1. **Stub reporter'ı değiştir** `trident/xai_text/small_llm_reporter.py` içinde
2. **Config'de etkinleştir**: `tasks.yml` içinde `SmallLLMReporter.enabled: true` ayarla  
3. **Çıktıları bağla**: Çıktı şemasına `report` alanı ekle
4. **Modelleri yükle**: CPU çıkarımı için kuantize modelleri kullan

```python
# Örnek entegrasyon
from report_llm import make_one_liner, build_inputs_for_llm

def generate_report(telemetry_data):
    prompt = build_inputs_for_llm(telemetry_data, style="resmi")
    one_liner = make_one_liner(MODEL_SYNC_PATH, prompt)
    # Asenkron rapor üretimini kuyruğa al...
    return one_liner
```

## 🚫 Sınırlamalar

- **Yalnızca model kapsamı**: `<angle>` yer tutucuları veya zamanlama enjeksiyonunu işlemez
- **Türkçe dil**: Yalnızca Türkçe teknik yazım için optimize edilmiştir
- **Telemetri-bağımlı**: Yapılandırılmış telemetri giriş formatı gerektirir
- **Operasyonel çıktı yok**: Kasıtlı olarak taktik/operasyonel dilden kaçınır

## 📊 Performans

- **Tek satır**: kısa çıktı (≤32 token)
- **Rapor**: orta uzunlukta çıktı (≤192 token)
- **Bellek**: model boyutuna bağlıdır; kuantize kopyalar RAM tasarrufu sağlar
- **Doğruluk**: Eğitim verisi kalitesi ve alan eşleşmesine bağlıdır

## 🤝 Katkıda Bulunma

1. Mevcut kod yapısı ve kalıplarını takip edin
2. Yeni işlevsellik için testler ekleyin
3. Türkçe dil çıktısı kalitesini sağlayın
4. Koruma politikalarının uygulandığını doğrulayın
5. Temsili telemetri verisi ile test edin

## 📄 Lisans

Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🆘 Destek

Şunlarla ilgili sorunlar için:
- **Eğitim/Çıkarım**: Model yollarını ve veri formatını kontrol edin
- **Korumalar başarısız oluyor**: Çıktının yasak içerik içermediğini doğrulayın  
- **Entegrasyon**: Telemetri verisinin beklenen şemaya uyduğundan emin olun
- **Performans**: Kuantize modelleri ve CPU optimizasyonunu kullanmayı düşünün

---

**Trident-XLM** - Çok modlu hedef tespit sistemleri için Türkçe açıklanabilirlik LLM'i.