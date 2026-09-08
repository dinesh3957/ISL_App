"""
Marathi → English Translator
Model  : AI4Bharat IndicTrans2 (indictrans2-indic-en-1B)  [best for Indian languages]
Fallback: Facebook mBART-large-50  (used if IndicTrans2 token is missing/invalid)
Features: High Quality Beam Search (num_beams=5) + Text Cleaning & Formatting
"""

import os
import re
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import torch
from pathlib import Path
from dotenv import load_dotenv, set_key

# ── Multi-threading CPU Optimizations ───────────────────────────────────────
torch.set_num_threads(os.cpu_count() or 4)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
ENV_FILE  = BASE_DIR / ".env"

# ── Load token from .env ──────────────────────────────────────────────────────
load_dotenv(ENV_FILE)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# ── Constants ─────────────────────────────────────────────────────────────────
INDIC_MODEL  = "ai4bharat/indictrans2-indic-en-1B"
MBART_MODEL  = "facebook/mbart-large-50-many-to-many-mmt"
SRC_INDIC    = "mar_Deva"
TGT_INDIC    = "eng_Latn"
SRC_MBART    = "mr_IN"
TGT_MBART    = "en_XX"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── Text Cleaner and Sentence Splitter ────────────────────────────────────────
def preprocess_marathi_text(text: str) -> list[str]:
    """Clean and split text properly so no Marathi words get missed."""
    # Add space after full stops if stuck to words (e.g. आहे.मराठी -> आहे. मराठी)
    text = re.sub(r'([।.?!\n])([^\s])', r'\1 \2', text)
    
    # Split by Marathi danda (।), English period (.), question mark (! ?), or newlines
    raw_sentences = re.split(r'(?<=[।.?!\n])\s+', text)
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else [text]


# ── Model Loaders ─────────────────────────────────────────────────────────────

def load_indictrans2(token: str):
    """Load IndicTrans2 model with HuggingFace token."""
    from huggingface_hub import login
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(
        INDIC_MODEL, trust_remote_code=True, token=token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDIC_MODEL,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    ip = IndicProcessor(inference=True)
    return tokenizer, model, ip, "indictrans2"


def load_mbart():
    """Load mBART fallback model (no token needed)."""
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL)
    model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model, None, "mbart"


def translate(text: str, tokenizer, model, ip, model_type: str) -> str:
    """High-quality translation with Beam Search (num_beams=5) for maximum accuracy."""
    if not text.strip():
        return ""

    sentences = preprocess_marathi_text(text)

    if model_type == "indictrans2":
        batch = ip.preprocess_batch(sentences, src_lang=SRC_INDIC, tgt_lang=TGT_INDIC)
        inputs = tokenizer(
            batch, padding="longest", truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,  # High accuracy Beam Search!
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_list = ip.postprocess_batch(decoded, lang=TGT_INDIC)
        return "\n".join(translated_list)

    else:  # mbart
        translated_list = []
        tokenizer.src_lang = SRC_MBART
        forced_bos = tokenizer.lang_code_to_id[TGT_MBART]

        for s in sentences:
            inputs = tokenizer(
                s, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(DEVICE)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=256,
                    num_beams=5,
                )
            translated_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return "\n".join(translated_list)


# ── Token Dialog ──────────────────────────────────────────────────────────────

class TokenDialog(tk.Toplevel):
    """Dialog to enter HuggingFace token and save it to .env."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("HuggingFace Token Setup")
        self.geometry("560x420")
        self.configure(bg="#0d0d1a")
        self.resizable(False, False)
        self.grab_set()   # modal
        self.result_token = None

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_skip)

    def _build(self):
        tk.Label(self, text="🔑  HuggingFace Token Setup",
                 font=("Segoe UI", 16, "bold"), bg="#0d0d1a", fg="#ff6b35"
                 ).pack(pady=(20, 5))

        tk.Label(self,
                 text="IndicTrans2 is the BEST Marathi AI — it needs a free HuggingFace token.\n"
                      "No charges. No subscriptions. Just a free account.",
                 font=("Segoe UI", 10), bg="#0d0d1a", fg="#a0a0c0",
                 justify="center", wraplength=500
                 ).pack(pady=(0, 15))

        # Steps
        steps_frame = tk.Frame(self, bg="#12122a", padx=15, pady=12)
        steps_frame.pack(fill="x", padx=20)

        steps = [
            ("Step 1", "Go to  https://huggingface.co  → Sign Up (FREE)"),
            ("Step 2", "Go to  https://huggingface.co/ai4bharat/indictrans2-indic-en-1B"),
            ("Step 3", "Click  'Request Access'  → Fill form → Wait few hours"),
            ("Step 4", "Go to  https://huggingface.co/settings/tokens  → New Token"),
            ("Step 5", "Paste your token below → Click Save"),
        ]
        for label, desc in steps:
            row = tk.Frame(steps_frame, bg="#12122a")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Segoe UI", 9, "bold"),
                     bg="#12122a", fg="#ff6b35", width=8, anchor="w"
                     ).pack(side="left")
            tk.Label(row, text=desc, font=("Segoe UI", 9),
                     bg="#12122a", fg="#c0c0e0", anchor="w"
                     ).pack(side="left")

        # Token input
        tk.Label(self, text="Paste your token here:",
                 font=("Segoe UI", 10, "bold"), bg="#0d0d1a", fg="#e0e0ff"
                 ).pack(pady=(15, 4))

        self.token_var = tk.StringVar()
        entry = tk.Entry(self, textvariable=self.token_var, show="*",
                         font=("Consolas", 11), bg="#12122a", fg="#ffffff",
                         insertbackground="#ff6b35", relief="flat",
                         width=48)
        entry.pack(padx=20, ipady=6)

        # Show/hide toggle
        self.show_token = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Show token", variable=self.show_token,
                       command=lambda: entry.config(show="" if self.show_token.get() else "*"),
                       bg="#0d0d1a", fg="#7070a0", selectcolor="#0d0d1a",
                       font=("Segoe UI", 9)
                       ).pack()

        # Buttons
        btn_frame = tk.Frame(self, bg="#0d0d1a")
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="✅  Save Token & Use IndicTrans2",
                  command=self._on_save,
                  bg="#ff6b35", fg="#ffffff", font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=16, pady=8, cursor="hand2"
                  ).pack(side="left", padx=(0, 10))

        tk.Button(btn_frame, text="⏭️  Skip — Use mBART Instead",
                  command=self._on_skip,
                  bg="#1a1a3e", fg="#a0a0c0", font=("Segoe UI", 10),
                  relief="flat", padx=12, pady=8, cursor="hand2"
                  ).pack(side="left")

        tk.Label(self,
                 text="Your token is saved ONLY on this PC in the .env file. Never shared.",
                 font=("Segoe UI", 8, "italic"), bg="#0d0d1a", fg="#505080"
                 ).pack()

    def _on_save(self):
        token = self.token_var.get().strip()
        if not token or token == "your_token_here":
            messagebox.showwarning("No Token", "Please paste a valid HuggingFace token.", parent=self)
            return
        if not token.startswith("hf_"):
            messagebox.showwarning("Invalid Token",
                                   "HuggingFace tokens start with 'hf_'\nPlease check and paste again.",
                                   parent=self)
            return
        # Save to .env
        set_key(str(ENV_FILE), "HF_TOKEN", token)
        self.result_token = token
        self.destroy()

    def _on_skip(self):
        self.result_token = None
        self.destroy()


# ── Main App ──────────────────────────────────────────────────────────────────

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi → English  |  AI4Bharat IndicTrans2 High-Accuracy Mode")
        self.root.geometry("820x670")
        self.root.configure(bg="#0d0d1a")
        self.root.resizable(True, True)

        self.tokenizer  = None
        self.model      = None
        self.ip         = None
        self.model_type = None
        self.hf_token   = HF_TOKEN

        self._build_ui()

        # Decide whether to show token dialog
        if not self.hf_token or self.hf_token == "your_token_here":
            self.root.after(300, self._show_token_dialog)
        else:
            self._load_model_async(use_indic=True)

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#12122a", pady=16)
        header.pack(fill="x")

        tk.Label(header, text="🇮🇳  Marathi → English Translator",
                 font=("Segoe UI", 22, "bold"), bg="#12122a", fg="#ff6b35"
                 ).pack()

        self.model_label_var = tk.StringVar(value="Initializing…")
        tk.Label(header, textvariable=self.model_label_var,
                 font=("Segoe UI", 9), bg="#12122a", fg="#7070a0"
                 ).pack(pady=(2, 0))

        device_txt = f"🖥️  {'GPU (CUDA) ⚡' if DEVICE == 'cuda' else f'CPU ({os.cpu_count() or 4} Cores) 🎯 (High Accuracy Mode: num_beams=5)'}"
        tk.Label(header, text=device_txt,
                 font=("Segoe UI", 9, "italic"), bg="#12122a", fg="#50c878"
                 ).pack(pady=(1, 0))

        # Status bar
        self.status_var = tk.StringVar(value="⏳  Starting up…")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 10, "italic"),
                 bg="#1a1a3e", fg="#f0e68c", anchor="w", padx=12, pady=6
                 ).pack(fill="x")

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x")
        self.progress.start(12)

        # Content
        content = tk.Frame(self.root, bg="#0d0d1a", padx=20, pady=15)
        content.pack(fill="both", expand=True)

        tk.Label(content, text="✍️  मराठी मजकूर टाइप करा  (Enter Marathi Text):",
                 font=("Segoe UI", 12, "bold"), bg="#0d0d1a", fg="#e0e0ff", anchor="w"
                 ).pack(fill="x", pady=(0, 5))

        self.input_text = scrolledtext.ScrolledText(
            content, height=8, font=("Nirmala UI", 15),
            bg="#12122a", fg="#ffffff", insertbackground="#ff6b35",
            relief="flat", padx=12, pady=10, wrap="word",
        )
        self.input_text.pack(fill="both", expand=True, pady=(0, 8))

        # Sample buttons
        sf = tk.Frame(content, bg="#0d0d1a")
        sf.pack(fill="x", pady=(0, 10))
        tk.Label(sf, text="Quick samples →", font=("Segoe UI", 9),
                 bg="#0d0d1a", fg="#7070a0").pack(side="left", padx=(0, 8))

        samples = [
            ("Sample 1", "नमस्ते, तुम्ही कसे आहात?"),
            ("Sample 2", "माझे नाव राहुल आहे आणि मी पुण्यात राहतो."),
            ("Sample 3", "महाराष्ट्र ही संतांची आणि शूरवीरांची भूमी आहे."),
            ("Sample 4", "छत्रपती शिवाजी महाराजांनी स्वराज्याची स्थापना केली."),
        ]
        for lbl, txt in samples:
            tk.Button(sf, text=lbl, command=lambda t=txt: self._insert_sample(t),
                      bg="#1a1a3e", fg="#a0c4ff", font=("Segoe UI", 9),
                      relief="flat", padx=8, pady=3, cursor="hand2"
                      ).pack(side="left", padx=(0, 5))

        # Translate button
        self.translate_btn = tk.Button(
            content, text="🔄  Translate to English",
            command=self._translate_async,
            bg="#ff6b35", fg="#ffffff", font=("Segoe UI", 13, "bold"),
            relief="flat", padx=20, pady=11,
            cursor="hand2", state="disabled",
        )
        self.translate_btn.pack(fill="x", pady=(0, 10))

        # Output
        tk.Label(content, text="📖  English Translation:",
                 font=("Segoe UI", 12, "bold"), bg="#0d0d1a", fg="#e0e0ff", anchor="w"
                 ).pack(fill="x", pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(
            content, height=6, font=("Segoe UI", 15),
            bg="#12122a", fg="#50c878",
            relief="flat", padx=12, pady=10, wrap="word", state="disabled",
        )
        self.output_text.pack(fill="both", expand=True)

        # Bottom
        bottom = tk.Frame(content, bg="#0d0d1a")
        bottom.pack(fill="x", pady=(8, 0))

        tk.Button(bottom, text="🔑 Change Token",
                  command=self._show_token_dialog,
                  bg="#1a1a3e", fg="#a0c4ff", font=("Segoe UI", 9),
                  relief="flat", padx=8, pady=4, cursor="hand2"
                  ).pack(side="left")

        tk.Button(bottom, text="🗑️  Clear",
                  command=self._clear_all,
                  bg="#1a1a3e", fg="#c0c0e0", font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=4, cursor="hand2"
                  ).pack(side="right")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _insert_sample(self, text):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)

    def _clear_all(self):
        self.input_text.delete("1.0", "end")
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
        self.status_var.set("✅  Ready.")

    def _show_token_dialog(self):
        dlg = TokenDialog(self.root)
        self.root.wait_window(dlg)
        if dlg.result_token:
            self.hf_token = dlg.result_token
            self.status_var.set("✅  Token saved! Loading IndicTrans2…")
            self.translate_btn.config(state="disabled")
            self.progress.pack(fill="x")
            self.progress.start(12)
            # Reload with IndicTrans2
            self._load_model_async(use_indic=True)
        else:
            # No token → use mBART if not already loaded
            if self.model_type != "mbart" and self.model is None:
                self._load_model_async(use_indic=False)

    # ── Model Loading ──────────────────────────────────────────────────────────
    def _load_model_async(self, use_indic: bool):
        self.status_var.set(
            "⏳  Loading IndicTrans2 model…" if use_indic
            else "⏳  Loading mBART model (~2.4 GB, first time only)…"
        )
        threading.Thread(
            target=self._load_model_worker,
            args=(use_indic,),
            daemon=True
        ).start()

    def _load_model_worker(self, use_indic: bool):
        try:
            if use_indic:
                tok, mdl, ip, mtype = load_indictrans2(self.hf_token)
            else:
                tok, mdl, ip, mtype = load_mbart()

            self.tokenizer  = tok
            self.model      = mdl
            self.ip         = ip
            self.model_type = mtype
            self.root.after(0, self._on_model_ready)

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda msg=err, indic=use_indic: self._on_model_error(msg, indic))

    def _on_model_ready(self):
        self.progress.stop()
        self.progress.pack_forget()
        if self.model_type == "indictrans2":
            self.model_label_var.set(
                "AI4Bharat IndicTrans2  ·  indictrans2-indic-en-1B  ·  High Accuracy Mode (num_beams=5)  ✅"
            )
            self.status_var.set("✅  IndicTrans2 loaded! Maximum Marathi translation quality enabled.")
        else:
            self.model_label_var.set(
                "Facebook mBART-large-50  ·  50 Languages  ·  No Login Required"
            )
            self.status_var.set("✅  mBART loaded! Enter Marathi text and click Translate.")
        self.translate_btn.config(state="normal")

    def _on_model_error(self, msg: str, was_indic: bool):
        self.progress.stop()
        self.progress.pack_forget()

        if "gated repo" in msg.lower() or "403" in msg or "restricted" in msg.lower():
            self.status_var.set("⚠️  IndicTrans2 access not approved yet — using mBART fallback")
            messagebox.showwarning(
                "IndicTrans2 Access Needed",
                "Your HuggingFace account does not have access to IndicTrans2 yet.\n\n"
                "Steps to get access (FREE):\n"
                "1. Go to: huggingface.co/ai4bharat/indictrans2-indic-en-1B\n"
                "2. Click 'Request Access'\n"
                "3. Wait a few hours for approval email\n\n"
                "⏭️  Switching to mBART model for now…"
            )
            self._load_model_async(use_indic=False)

        elif "invalid token" in msg.lower() or "401" in msg or "credentials" in msg.lower():
            self.status_var.set("❌  Invalid token — please re-enter")
            if messagebox.askyesno("Invalid Token",
                                   "The HuggingFace token is invalid.\n\n"
                                   "Do you want to enter a new token?"):
                self._show_token_dialog()
            else:
                self._load_model_async(use_indic=False)

        else:
            self.status_var.set(f"❌  Error loading model: {msg[:80]}")
            if was_indic:
                if messagebox.askyesno("Model Load Failed",
                                       f"IndicTrans2 failed to load:\n{msg}\n\n"
                                       "Switch to mBART fallback model instead?"):
                    self._load_model_async(use_indic=False)
            else:
                messagebox.showerror("Error", f"Could not load any model:\n{msg}")

    # ── Translation ───────────────────────────────────────────────────────────
    def _translate_async(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter Marathi text first.")
            return

        self.translate_btn.config(state="disabled", text="⏳  Translating (High Quality Mode)…")
        self.status_var.set("🔄  Translating with High Accuracy (num_beams=5)…")

        def worker():
            try:
                result = translate(text, self.tokenizer, self.model, self.ip, self.model_type)
                self.root.after(0, lambda r=result: self._show_result(r))
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._show_translate_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, result: str):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", result)
        self.output_text.config(state="disabled")
        self.translate_btn.config(state="normal", text="🔄  Translate to English")
        self.status_var.set("✅  High-quality translation complete!")

    def _show_translate_error(self, msg: str):
        self.translate_btn.config(state="normal", text="🔄  Translate to English")
        self.status_var.set(f"❌  Translation error")
        messagebox.showerror("Translation Error",
                             f"Translation failed:\n{msg}\n\n"
                             "Please try again or restart the app.")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
