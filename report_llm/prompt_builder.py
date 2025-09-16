from .types import TelemetryNLIn, StylePreset, Contribution

def _fmt_contrib(c: Contribution) -> str:
        sgn = "+" if c.value_pct >= 0 else "-"
        return f"{c.name} | {c.modality} | {sgn}{abs(c.value_pct):.2f} | {c.note}"

def build_inputs_for_llm(t: TelemetryNLIn, style: StylePreset = "resmi") -> str:
        """Deterministically serialize telemetry into a model-facing prompt.
        IMPORTANT: This contains ONLY telemetry facts from the model pipeline.
        No angle-bracket placeholders, IDs, timestamps, or wrapper-provided fields appear here.
        """
        pos = [c for c in t.contributions if c.sign == "pos"]
        neg = [c for c in t.contributions if c.sign == "neg"]

        pos_lines = "\n".join(f"- {_fmt_contrib(c)}" for c in pos) if pos else "- (yok)"
        neg_lines = "\n".join(f"- {_fmt_contrib(c)}" for c in neg) if neg else "- (yok)"

        # flags/exp/meta are inserted as raw dicts; upstream must ensure they are JSON-safe
        return f"""Görev: Aşağıdaki telemetri ve açıklanabilirlik özetinden yola çıkarak
3.0 ve 4.0 bölümlerinin TÜM metnini sıfırdan yaz.
Sayıları değiştirme; yeni sayı uydurma; taktiksel/operasyonel tavsiye verme.
Üslup: {style}. Dil: Türkçe. Hedef kitle: yazılım/sistem mühendisleri.

[MODEL GİRİŞ ÖZETİ]
- p_hit_calib={t.p_hit_calib:.2f}, p_kill_calib={t.p_kill_calib:.2f},
    p_hit_masked={t.p_hit_masked:.2f}, p_kill_masked={t.p_kill_masked:.2f}, spoof_risk={t.spoof_risk:.2f}
- FLAGS: {t.flags}
- EXPL: {t.exp}
- META: {t.meta}

[SKOR KATKILARI / POZİTİF]
{pos_lines}

[SKOR KATKILARI / NEGATİF]
{neg_lines}

Yazım talimatları:
- 3.0 “Birincil Gerekçe” için 3-5 cümlelik tutarlı bir anlatım üret.
- 4.0 “Karara Etki Eden Faktörler” bölümünde pozitif/negatif unsurları
    metinle *anlatarak* özetle (listeleri yorumla); sayıları değiştirme.
- “SpoofShield”, “Doppler”, “RCS”, “LWIR” gibi teknik adlar varsa bağlama göre doğal biçimde kullan.
- Yasak: “ateş et”, “nişan al” vb. operasyonel ifadeler (sadece yazılım değerlendirmesi).
"""