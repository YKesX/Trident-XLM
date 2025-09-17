#!/usr/bin/env python3
"""
Simple Tkinter GUI for Trident-XLM
- Enter telemetry by hand (JSON) or generate random
- Runs one-liner (Flan-T5) and report (mT5)
- Saves outputs under reports/ as .txt files automatically
"""
import os, sys, json, types, importlib.machinery, datetime, tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import asdict

# Project imports
from report_llm.types import TelemetryNLIn, Contribution
from report_llm.prompt_builder import build_inputs_for_llm
from report_llm.summarizer_sync import make_one_liner
from report_llm.summarizer_async import make_report

# Torch-only hardening (avoid TF issues)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TORCH", "1")
if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow"); _tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None); sys.modules["tensorflow"] = _tf
if "ml_dtypes" not in sys.modules:
    _mld = types.ModuleType("ml_dtypes"); _mld.__spec__ = importlib.machinery.ModuleSpec("ml_dtypes", loader=None); sys.modules["ml_dtypes"] = _mld

def _poor(txt: str) -> bool:
    if not txt: return True
    t = txt.strip(); return (len(t) < 40) or (t.count(" ") < 5)

def fallback_one_liner(t: TelemetryNLIn) -> str:
    return (
        f"Kalibre güven skorları (vuruş={t.p_hit_calib:.2f}, imha={t.p_kill_calib:.2f}); "
        f"maske sonrası (vuruş={t.p_hit_masked:.2f}, imha={t.p_kill_masked:.2f}). "
        f"Sahtecilik riski {t.spoof_risk:.2f}."
    )

def fallback_report(t: TelemetryNLIn) -> str:
    pos = [c for c in t.contributions if c.sign == "pos"]
    neg = [c for c in t.contributions if c.sign == "neg"]
    s = [
        "3.0. Birincil Gerekçe",
        "Sistem, çoklu sensör kaynaklarından gelen tutarlı sinyalleri değerlendirerek yüksek güven seviyesine ulaşmıştır.",
        "4.0. Karara Etki Eden Faktörler",
        ("Pozitif katkı: " + ", ".join(f"{c.name} (+{c.value_pct:.2f}%, {c.modality})" for c in pos)) if pos else "Pozitif katkı yok.",
        ("Negatif katkı: " + ", ".join(f"{c.name} ({c.value_pct:.2f}%, {c.modality})" for c in neg)) if neg else "Negatif katkı yok.",
    ]
    return "\n".join(s)

def gen_random_telem() -> TelemetryNLIn:
    # Reuse logic from random_demo but minimal inline to avoid circular imports
    import random
    from typing import List
    SENSOR_MODES = ["RGB", "IR", "RGB+IR", "RADAR+EO"]
    ATTN_SPOTS = ["burun", "gövde", "orta-bölüm", "arka", "sol-kanat", "sağ-kanat", "iz"]
    POS = [
        ("Radar Kesit Alanı (RCS)", "RADAR (Ka-band)", 15, 45, "güçlü yansıma"),
        ("Termal İmza Analizi", "EO (LWIR)", 10, 40, "yüksek sıcaklık stabil"),
        ("Doppler Tehdit", "RADAR (Doppler)", 8, 30, "yaklaşma vektörü tutarlı"),
    ]
    NEG = [
        ("SpoofShield Düzeltmesi", "Guard", -2, -0.3, "tutarlı sensörler"),
        ("Atmosferik Parıldama", "EO", -3, -0.5, "uzun menzil etkisi"),
        ("Parazit/Karıştırma", "RADAR", -3, -0.7, "düşük arka plan"),
    ]
    contribs = []
    for name, mod, lo, hi, note in POS:
        if random.random() < 0.8:
            from report_llm.types import Contribution
            contribs.append(Contribution(name, mod, "pos", round(random.uniform(lo, hi), 2), note))
    for name, mod, lo, hi, note in NEG:
        if random.random() < 0.7:
            from report_llm.types import Contribution
            contribs.append(Contribution(name, mod, "neg", round(random.uniform(hi, lo), 2), note))
    p_hit_calib = round(random.uniform(0.6, 0.99), 2)
    p_kill_calib = round(random.uniform(0.5, p_hit_calib), 2)
    p_hit_masked = max(0.0, min(1.0, round(p_hit_calib - random.uniform(0.0, 0.05), 2)))
    p_kill_masked = max(0.0, min(1.0, round(p_kill_calib - random.uniform(0.0, 0.05), 2)))
    t = TelemetryNLIn(
        p_hit_calib=p_hit_calib,
        p_kill_calib=p_kill_calib,
        p_hit_masked=p_hit_masked,
        p_kill_masked=p_kill_masked,
        spoof_risk=round(random.uniform(0.0, 0.2), 2),
        flags={"mask_applied": random.random() < 0.5, "roi_coverage": round(random.uniform(0.6, 0.95), 2), "blur": round(random.uniform(0.0, 0.2), 2)},
        exp={"attn_hotspots": random.sample(ATTN_SPOTS, k=random.randint(1, 3)), "gradcam_roi": random.choice(["merkez", "sol", "sağ", "üst", "alt"])},
        meta={"sensor_mode": random.choice(SENSOR_MODES), "approach": random.choice(["yaklaşan", "uzaklaşan", "sabit", "manzara"]), "speed_kph": round(random.uniform(10, 60), 1)},
        contributions=contribs,
    )
    return t

class TridentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trident-XLM GUI")
        self.geometry("900x700")
        self._apply_dark_theme()
        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Model paths
        mp = ttk.LabelFrame(frm, text="Model Paths")
        mp.pack(fill=tk.X, pady=6)
        ttk.Label(mp, text="One-liner (Flan):").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.flan_var = tk.StringVar(value=os.environ.get("FLAN_MODEL", "report_llm/exports/flan_one_liner"))
        ttk.Entry(mp, textvariable=self.flan_var, width=80).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(mp, text="Browse", command=lambda: self._browse(self.flan_var)).grid(row=0, column=2, padx=4)

        ttk.Label(mp, text="Report (mT5):").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.mt5_var = tk.StringVar(value=os.environ.get("MT5_MODEL", "report_llm/exports/mt5_report"))
        ttk.Entry(mp, textvariable=self.mt5_var, width=80).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(mp, text="Browse", command=lambda: self._browse(self.mt5_var)).grid(row=1, column=2, padx=4)

        # Input area
        inp = ttk.LabelFrame(frm, text="Telemetry Input (JSON) ")
        inp.pack(fill=tk.BOTH, expand=True, pady=6)
        self.text = tk.Text(inp, height=16, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Controls
        ctrl = ttk.Frame(frm)
        ctrl.pack(fill=tk.X)
        self.style_var = tk.StringVar(value="resmi")
        ttk.Label(ctrl, text="Üslup:").pack(side=tk.LEFT)
        ttk.Combobox(ctrl, textvariable=self.style_var, values=["resmi", "madde", "anlatımcı"], width=12, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Rastgele Doldur", command=self.fill_random).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Dosyadan Yükle", command=self.load_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Çalıştır ve Kaydet", command=self.run_and_save).pack(side=tk.RIGHT, padx=6)

        # Output area
        out = ttk.LabelFrame(frm, text="Çıktılar")
        out.pack(fill=tk.BOTH, expand=True, pady=6)
        self.one_var = tk.StringVar()
        self.rep_text = tk.Text(out, height=10, wrap=tk.WORD)
        ttk.Label(out, textvariable=self.one_var, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=4)
        self.rep_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Apply dark styling to text widgets (after creation)
        try:
            entrybg = '#2a2d2e'; fg = '#e5e5e5'; selbg = '#264f78'; bg = '#1e1e1e'
            for widget in (self.text, self.rep_text):
                widget.configure(bg=entrybg, fg=fg, insertbackground=fg,
                                 selectbackground=selbg, selectforeground=fg,
                                 highlightbackground=bg, highlightcolor=bg)
        except Exception:
            pass

    def _browse(self, var: tk.StringVar):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def fill_random(self):
        t = gen_random_telem()
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, json.dumps({"inputs": asdict(t)}, ensure_ascii=False, indent=2))

    def load_file(self):
        p = filedialog.askopenfilename(filetypes=[("JSON/JSONL", "*.json;*.jsonl"), ("All", "*.*")])
        if not p: return
        with open(p, "r", encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                line = f.readline().strip(); data = json.loads(line) if line else {}
            else:
                data = json.load(f)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, json.dumps(data, ensure_ascii=False, indent=2))

    def _parse_telem(self) -> TelemetryNLIn:
        raw = self.text.get("1.0", tk.END).strip()
        if not raw:
            raise ValueError("Girdi boş.")
        data = json.loads(raw)
        inputs = data.get("inputs") or data
        contribs = [Contribution(**c) for c in inputs.get("contributions", [])]
        return TelemetryNLIn(
            p_hit_calib=inputs["p_hit_calib"],
            p_kill_calib=inputs["p_kill_calib"],
            p_hit_masked=inputs["p_hit_masked"],
            p_kill_masked=inputs["p_kill_masked"],
            spoof_risk=inputs["spoof_risk"],
            flags=inputs.get("flags", {}),
            exp=inputs.get("exp", {}),
            meta=inputs.get("meta", {}),
            contributions=contribs,
        )

    def run_and_save(self):
        try:
            telem = self._parse_telem()
            prompt = build_inputs_for_llm(telem, style=self.style_var.get())
            flan = self.flan_var.get().strip(); mt5 = self.mt5_var.get().strip()

            # Generate
            try:
                one = make_one_liner(flan, prompt)
            except Exception as e:
                one = fallback_one_liner(telem)
            if _poor(one):
                one = fallback_one_liner(telem)

            try:
                rep = make_report(mt5, prompt)
            except Exception as e:
                rep = fallback_report(telem)
            if _poor(rep):
                rep = fallback_report(telem)

            # Show
            self.one_var.set("ONE-LINER: " + one)
            self.rep_text.delete("1.0", tk.END); self.rep_text.insert(tk.END, rep)

            # Save under reports/
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("reports", exist_ok=True)
            base = os.path.join("reports", f"report_{ts}")
            with open(base + "_one_liner.txt", "w", encoding="utf-8") as f:
                f.write(one)
            with open(base + "_report.txt", "w", encoding="utf-8") as f:
                f.write(rep)
            messagebox.showinfo("Kaydedildi", f"Kayıt: {base}_*.txt")

        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def _apply_dark_theme(self):
        try:
            style = ttk.Style()
            # Use a theme that allows custom colors
            try:
                style.theme_use('clam')
            except Exception:
                pass
            bg = '#1e1e1e'
            fg = '#e5e5e5'
            btn_bg = '#2d2d2d'
            entry_bg = '#2a2d2e'
            accent = '#0e639c'

            self.configure(bg=bg)
            # Base
            style.configure('.', background=bg, foreground=fg)
            # Frames and labels
            style.configure('TFrame', background=bg)
            style.configure('TLabelframe', background=bg)
            style.configure('TLabelframe.Label', background=bg, foreground=fg)
            style.configure('TLabel', background=bg, foreground=fg)
            # Buttons
            style.configure('TButton', background=btn_bg, foreground=fg, focuscolor=accent)
            style.map('TButton', background=[('active', '#3a3a3a')])
            # Entry/Combobox fields
            style.configure('TEntry', fieldbackground=entry_bg, foreground=fg)
            style.configure('TCombobox', fieldbackground=entry_bg, foreground=fg, background=entry_bg)
            style.map('TCombobox', fieldbackground=[('readonly', entry_bg)],
                      foreground=[('readonly', fg)],
                      background=[('active', entry_bg)])
        except Exception:
            # Fallback: ignore styling errors
            pass

if __name__ == "__main__":
    app = TridentGUI()
    app.mainloop()
