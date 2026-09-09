"""
Marathi → English Translator — Gradio App
Hugging Face Spaces deployment with ZeroGPU support
Engine  : CTranslate2 INT8 → PyTorch IndicTrans2 → mBART fallback
Features: PDF upload, real-time translation, professional UI
"""

import os
import re
import io
import time
import torch
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# ZeroGPU support on Hugging Face
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                return lambda f: f
            return func


# ── Paths & Token ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CT2_DIR  = BASE_DIR / "ct2_model"
load_dotenv(BASE_DIR / ".env")

HF_TOKEN    = (os.environ.get("HF_TOKEN") or "").strip()
INDIC_MODEL = "ai4bharat/indictrans2-indic-en-1B"
MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
SRC_INDIC   = "mar_Deva"
TGT_INDIC   = "eng_Latn"
SRC_MBART   = "mr_IN"
TGT_MBART   = "en_XX"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(os.cpu_count() or 4)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    text = re.sub(r'([।.?!\n])([^\s])', r'\1 \2', text)
    raw  = re.split(r'(?<=[।.?!\n])\s+', text)
    out  = [s.strip() for s in raw if s.strip()]
    return out if out else [text.strip()]


def read_pdf(file_path: str) -> str:
    try:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t and t.strip():
                    pages.append(t.strip())
        return "\n\n".join(pages)
    except Exception as e:
        return f"[PDF Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING (lazy — load once on first translation)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = {}

def get_model():
    if "engine" in _model_cache:
        return _model_cache["engine"], _model_cache["tokenizer"], _model_cache["ip"], _model_cache["type"], _model_cache["label"]

    token = HF_TOKEN

    # Try CTranslate2 first
    if token and token.startswith("hf_"):
        try:
            import ctranslate2
            from huggingface_hub import login, snapshot_download
            from transformers import AutoTokenizer
            from IndicTransToolkit import IndicProcessor

            login(token=token, add_to_git_credential=False)

            if not CT2_DIR.exists() or not (CT2_DIR / "model.bin").exists():
                model_path = snapshot_download(
                    repo_id=INDIC_MODEL, token=token,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
                )
                converter = ctranslate2.converters.OpusMTConverter(model_path)
                converter.convert(str(CT2_DIR), quantization="int8", force=True)

            translator = ctranslate2.Translator(
                str(CT2_DIR),
                device=DEVICE,
                inter_threads=2,
                intra_threads=os.cpu_count() or 4,
            )
            tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL, trust_remote_code=True, token=token)
            ip = IndicProcessor(inference=True)
            _model_cache.update({"engine": translator, "tokenizer": tokenizer, "ip": ip, "type": "ct2", "label": "⚡ CTranslate2 INT8"})
            return translator, tokenizer, ip, "ct2", "⚡ CTranslate2 INT8"
        except Exception:
            pass

        # Try PyTorch IndicTrans2
        try:
            from huggingface_hub import login
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit import IndicProcessor

            login(token=token, add_to_git_credential=False)
            tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL, trust_remote_code=True, token=token)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                INDIC_MODEL, trust_remote_code=True, token=token, torch_dtype=torch.float32
            ).to(DEVICE)
            model.eval()
            ip = IndicProcessor(inference=True)
            _model_cache.update({"engine": model, "tokenizer": tokenizer, "ip": ip, "type": "pytorch_indic", "label": "🔵 IndicTrans2"})
            return model, tokenizer, ip, "pytorch_indic", "🔵 IndicTrans2"
        except Exception:
            pass

    # Fallback: mBART
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL)
    model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL).to(DEVICE)
    model.eval()
    _model_cache.update({"engine": model, "tokenizer": tokenizer, "ip": None, "type": "mbart", "label": "🟡 mBART-50"})
    return model, tokenizer, None, "mbart", "🟡 mBART-50"


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSLATE FUNCTION (Runs on ZeroGPU with 15s limit & CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

@spaces.GPU(duration=15)
def translate(marathi_text: str, pdf_file):

    # Use PDF text if uploaded
    if pdf_file is not None:
        marathi_text = read_pdf(pdf_file.name)
        if marathi_text.startswith("[PDF Error"):
            return marathi_text, "❌ PDF read failed", ""

    if not marathi_text or not marathi_text.strip():
        return "", "⚠️ Please enter Marathi text or upload a PDF.", ""

    t0 = time.time()
    engine, tokenizer, ip, model_type, model_label = get_model()
    sentences = split_sentences(marathi_text)
    results   = []

    curr_device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type != "ct2" and hasattr(engine, "to"):
        engine.to(curr_device)

    try:
        if model_type == "ct2":
            for s in sentences:
                batch   = ip.preprocess_batch([s], src_lang=SRC_INDIC, tgt_lang=TGT_INDIC)
                tokens  = [tokenizer.tokenize(sent) for sent in batch]
                output  = engine.translate_batch(tokens, beam_size=4, max_decoding_length=256)
                decoded = [tokenizer.convert_tokens_to_string(r.hypotheses[0]) for r in output]
                results.extend(ip.postprocess_batch(decoded, lang=TGT_INDIC))

        elif model_type == "pytorch_indic":
            for s in sentences:
                batch  = ip.preprocess_batch([s], src_lang=SRC_INDIC, tgt_lang=TGT_INDIC)
                inputs = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(curr_device)
                with torch.inference_mode():
                    out = engine.generate(**inputs, max_length=256, num_beams=4)
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                results.extend(ip.postprocess_batch(decoded, lang=TGT_INDIC))

        else:  # mbart
            tokenizer.src_lang = SRC_MBART
            forced_bos = tokenizer.lang_code_to_id[TGT_MBART]
            for s in sentences:
                inputs = tokenizer(s, return_tensors="pt", padding=True, truncation=True, max_length=256).to(curr_device)
                with torch.inference_mode():
                    out = engine.generate(**inputs, forced_bos_token_id=forced_bos, max_length=256, num_beams=4)
                results.append(tokenizer.decode(out[0], skip_special_tokens=True))

        elapsed     = time.time() - t0
        final_text  = "\n\n".join(results)
        word_count  = len(final_text.split())
        status_msg  = f"✅ Done in {elapsed:.1f}s · {word_count} words · {model_label} · {curr_device.upper()}"
        return final_text, status_msg, marathi_text


    except Exception as e:
        return "", f"❌ Error: {str(e)}", ""


def clear_all():
    return "", None, "", ""


# ─────────────────────────────────────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

:root {
    --bg: #0f1117;
    --card: #1a1d27;
    --input: #13151f;
    --accent: #5865f2;
    --accent2: #7983f5;
    --success: #3ba55d;
    --text: #e8eaf0;
    --muted: #8b8fa8;
    --border: rgba(255,255,255,0.07);
    --radius: 12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

/* Nav */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
    border-radius: 0;
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #ff7043, #e64a19);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.nav-title { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.nav-badge {
    font-size: 0.72rem; padding: 4px 12px; border-radius: 20px;
    background: rgba(59,165,93,0.12); color: #3ba55d;
    border: 1px solid rgba(59,165,93,0.3); font-weight: 500;
}

/* Heading */
.page-head { text-align: center; padding: 4px 0 20px; }
.page-head h1 {
    font-size: 2rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.03em; margin: 0 0 6px;
}
.page-head p { font-size: 0.88rem; color: var(--muted); margin: 0; }

/* Stats */
.stats-row { display: flex; gap: 10px; margin-bottom: 20px; }
.stat { flex: 1; background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: 13px; text-align: center; }
.stat-v { font-size: 1.25rem; font-weight: 700; color: var(--accent2); display: block; }
.stat-l { font-size: 0.67rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }

/* Inputs */
textarea, input[type="text"] {
    background: var(--input) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans Devanagari', 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88,101,242,0.2) !important;
    outline: none !important;
}

/* Buttons */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent), #4752c4) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 24px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(88,101,242,0.35) !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    box-shadow: 0 6px 20px rgba(88,101,242,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Labels */
label span, .label-wrap span {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Output */
.output-area textarea {
    color: #cde8d5 !important;
    border-left: 3px solid var(--success) !important;
    background: var(--input) !important;
}

/* Status */
.status-bar textarea {
    font-size: 0.85rem !important;
    color: var(--muted) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}

/* Tabs */
.tab-nav { background: var(--card) !important; border-bottom: 1px solid var(--border) !important; }
.tab-nav button { color: var(--muted) !important; }
.tab-nav button.selected { color: var(--accent2) !important; border-bottom: 2px solid var(--accent2) !important; }

/* Samples */
.sample-btn { background: var(--card) !important; border: 1px solid var(--border) !important;
               color: var(--muted) !important; border-radius: 20px !important;
               font-size: 0.8rem !important; padding: 5px 14px !important; }
.sample-btn:hover { border-color: var(--accent) !important; color: var(--accent2) !important; }

/* Footer */
.footer {
    text-align: center; font-size: 0.74rem; color: var(--muted);
    border-top: 1px solid var(--border); padding: 16px 0; margin-top: 24px;
}

/* File upload */
.upload-btn { background: var(--input) !important; border: 1.5px dashed var(--border) !important;
               border-radius: 8px !important; }

/* Block labels */
.block.padded { background: transparent !important; }
"""

SAMPLES = [
    "नमस्ते, तुम्ही कसे आहात?",
    "माझे नाव राहुल आहे आणि मी पुण्यात राहतो.",
    "महाराष्ट्र ही संतांची आणि शूरवीरांची भूमी आहे.",
    "छत्रपती शिवाजी महाराजांनी स्वराज्याची स्थापना केली.",
]

with gr.Blocks(title="Marathi → English Translator") as demo:

    # Navigation Bar
    gr.HTML("""
    <div class="nav-bar">
        <div class="nav-left">
            <div class="nav-icon">🇮🇳</div>
            <span class="nav-title">भाषांतर &nbsp;·&nbsp; Translator</span>
        </div>
        <span class="nav-badge">● AI Powered · Free · Fast</span>
    </div>
    """)

    # Page Title
    gr.HTML("""
    <div class="page-head">
        <h1>Marathi → English</h1>
        <p>AI-powered translation · IndicTrans2 · CTranslate2 INT8 · Free forever</p>
    </div>
    """)

    # Stats Row
    gr.HTML("""
    <div class="stats-row">
        <div class="stat"><span class="stat-v">⚡</span><span class="stat-l">Fast on GPU</span></div>
        <div class="stat"><span class="stat-v">📄</span><span class="stat-l">PDF Support</span></div>
        <div class="stat"><span class="stat-v">★★★</span><span class="stat-l">High Quality</span></div>
        <div class="stat"><span class="stat-v">∞</span><span class="stat-l">No Limits</span></div>
    </div>
    """)

    # Main layout — two columns
    with gr.Row(equal_height=False):

        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        with gr.Column(scale=1):

            with gr.Tab("✍️  Type / Paste"):
                marathi_input = gr.Textbox(
                    label="Input — Marathi Text",
                    placeholder="मराठी मजकूर येथे लिहा किंवा पेस्ट करा…\n\nType or paste Marathi text here…",
                    lines=10,
                    max_lines=30,
                )
                gr.HTML('<p style="font-size:0.78rem;color:#555870;margin:6px 0 8px">Quick samples:</p>')
                with gr.Row():
                    for i, sample in enumerate(SAMPLES):
                        short = ["Greeting", "Intro", "Maharashtra", "History"][i]
                        btn = gr.Button(short, elem_classes=["sample-btn"], size="sm")
                        btn.click(fn=lambda s=sample: s, outputs=marathi_input)

            with gr.Tab("📄  Upload PDF"):
                pdf_input = gr.File(
                    label="Upload Marathi PDF",
                    file_types=[".pdf"],
                    elem_classes=["upload-btn"],
                )
                gr.HTML('<p style="font-size:0.78rem;color:#555870;margin:6px 0">PDF text will be extracted and translated automatically.</p>')

            with gr.Row():
                translate_btn = gr.Button("Translate →", variant="primary", size="lg")
                clear_btn     = gr.Button("Clear", size="lg")

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        with gr.Column(scale=1):
            english_output = gr.Textbox(
                label="Translation — English",
                placeholder="Translation will appear here…",
                lines=14,
                max_lines=40,
                interactive=False,
                elem_classes=["output-area"],
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1,
                elem_classes=["status-bar"],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer">
        भाषांतर · Marathi to English Translator &nbsp;|&nbsp;
        AI4Bharat IndicTrans2 · CTranslate2 INT8 &nbsp;|&nbsp;
        🆓 Free · No daily limits
    </div>
    """)

    # ── Wire up events ────────────────────────────────────────────────────────
    pdf_state = gr.State(None)

    translate_btn.click(
        fn=translate,
        inputs=[marathi_input, pdf_input],
        outputs=[english_output, status_output, marathi_input],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[marathi_input, pdf_input, english_output, status_output],
    )


if __name__ == "__main__":
    print("🚀 Pre-loading model to cache...")
    try:
        get_model()
        print("✅ Model cached successfully!")
    except Exception as e:
        print(f"⚠️ Pre-load warning: {e}")

    demo.launch(css=CUSTOM_CSS)


