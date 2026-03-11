# ---------------------------------------------------------
# Italian Speaking Practice — Claude Sonnet Edition (Optimized)
# - Single provider: Anthropic Claude Sonnet (no LanguageTool/OpenAI)
# - Token optimizations: compact prompt & JSON, output caps, cleanup
# - 60s limit: recordings >60s rejected; uploads truncated to 60s for analysis
# - Multi-format uploader (WAV/MP3/M4A/OGG/WEBM)
# - Word timestamps + low-confidence highlighting (ASR: faster-whisper)
# - Optional TTS for corrected text (gTTS, Italian)
# ---------------------------------------------------------

import os
import io
import re
import time
import json
import wave
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from gtts import gTTS
import requests  # only used for mic/uploader fallback if needed

# Anthropic (Claude) SDK
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# ---------------------------
# ---------- Config ----------
# ---------------------------

MAX_RECORD_SECONDS = 60.0  # hard limit for recordings
TRUNCATE_UPLOAD_SECONDS = 60.0  # analyze only the first 60s from uploads
DEFAULT_SONNET_MODEL = "claude-3-5-sonnet-latest"  # change if your tenant exposes a newer snapshot

PROMPTS = [
    "Describe your ideal day from start to finish.",
    "Tell us about a trip that changed your life.",
    "Explain a typical dish from your region and how to prepare it.",
    "Discuss a social issue in Italy and propose some solutions.",
    "Compare life in the city and the countryside: pros and cons.",
    "If you could change one law, which would it be and why?",
    "Talk about a book or film that impacted you and explain why."
]

RUBRIC_HINT = "A2–B1 orale: correzioni essenziali, spiegazioni brevi, tono incoraggiante."

# ---------------------------
# ---------- Data -----------
# ---------------------------

@dataclass
class Issue:
    msg: str
    start: int
    end: int
    suggestion: str
    rule_id: str
    category: str = "Custom"

# ---------------------------
# -------- Utilities --------
# ---------------------------

def warning_banner():
    st.info(
        "⏱️ **Recording limit**: up to 60 seconds. "
        "Uploads longer than 60s will be **analyzed only for the first 60s**."
    )

def get_api_key() -> Optional[str]:
    return (st.secrets.get("ANTHROPIC_API_KEY")
            if "ANTHROPIC_API_KEY" in st.secrets
            else os.getenv("ANTHROPIC_API_KEY"))

@st.cache_resource(show_spinner=False)
def get_anthropic_client(api_key: str):
    return Anthropic(api_key=api_key)

@st.cache_resource(show_spinner=False)
def load_fw_model(name: str, compute: str) -> WhisperModel:
    return WhisperModel(name, device="cpu", compute_type=compute)

def wav_duration_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            return frames / float(rate)
    except wave.Error:
        return 0.0

def save_bytes_to_file(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        f.write(data)
    return tmp.name

# -------- Text normalization & compacting ----------

MULTISPACE = re.compile(r"\s{2,}")
FILLERS = {"ehm", "eh", "mmm", "uh", "cioè", "tipo", "diciamo", "boh", "praticamente"}

def normalize_italian_text(text: str) -> str:
    # merge l ' amico -> l'amico, and cleanup apostrophes/whitespace
    t = re.sub(r"\b([lLdDaAeEoOuU]l?)\s*'\s*([a-zA-Zàèéìòóù])", r"\1'\2", text)
    t = t.replace(" ’ ", "’").replace(" ' ", "'")
    t = t.replace(" ’", "’").replace("’ ", "’")
    t = t.replace(" '", "'").replace("' ", "'")
    t = MULTISPACE.sub(" ", t)
    return t.strip()

def remove_fillers_tokens(text: str) -> str:
    toks = [w for w in text.split() if w.strip()]
    kept = []
    for w in toks:
        w_clean = w.strip(".,;:!?\"'()[]{}").lower()
        if w_clean in FILLERS:
            continue
        kept.append(w)
    return " ".join(kept)

def dedupe_repetitions(text: str) -> str:
    # remove exact adjacent repetition of short phrases (simple heuristic)
    parts = text.split()
    out = []
    for w in parts:
        if not out or out[-1].lower() != w.lower():
            out.append(w)
    return " ".join(out)

def compact_for_llm(text: str, remove_fillers: bool = True, max_words: int = 180) -> str:
    t = normalize_italian_text(text)
    if remove_fillers:
        t = remove_fillers_tokens(t)
    t = dedupe_repetitions(t)
    # cap by words (rough token proxy)
    words = t.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).strip()

# ---------------------------
# ---- Transcription w/ timestamps
# ---------------------------

def transcribe(model: WhisperModel, audio_path: str):
    """
    Transcribe Italian with word timestamps.
    Returns: (full_text, words, seg_stats)
    """
    temperature = [0.0, 0.2, 0.4]
    segments, _ = model.transcribe(
        audio_path,
        language="it",
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        best_of=5,
        temperature=temperature,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt=("Trascrivi in italiano standard senza riformulare. "
                        "Evita correzioni non necessarie. Non tradurre."),
        condition_on_previous_text=False,
        word_timestamps=True,
        without_timestamps=False
    )

    words = []
    seg_stats = []
    texts = []

    for s in segments:
        if s.text:
            texts.append(s.text.strip())

        if getattr(s, "words", None):
            for w in s.words:
                avg = s.avg_logprob if s.avg_logprob is not None else -5.0
                conf = max(min((avg + 5) / 5, 1.0), 0.0)  # 0..1 heuristic
                words.append({
                    "word": w.word,
                    "start": float(getattr(w, "start", 0.0) or 0.0),
                    "end": float(getattr(w, "end", 0.0) or 0.0),
                    "conf": float(conf)
                })

        seg_stats.append({
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end": float(getattr(s, "end", 0.0) or 0.0),
            "avg_logprob": float(getattr(s, "avg_logprob", -5.0)),
            "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
        })

    full_text = " ".join(texts).strip()
    return full_text, words, seg_stats

def truncate_words_to_seconds(words: List[Dict], max_seconds: float) -> List[Dict]:
    return [w for w in words if w.get("end", 0.0) <= max_seconds]

def words_to_text(words: List[Dict]) -> str:
    return " ".join([w["word"] for w in words])

# ---------------------------
# ---- Claude Sonnet Feedback
# ---------------------------

def build_compact_prompt(level_hint: str) -> str:
    # Minimal, reusable system-style content
    return (
        "Sei un correttore per studenti di italiano (A2–B1). "
        "Correggi solo errori chiari e frequenti, senza riscrivere troppo. "
        "Usa spiegazioni brevi e semplici. "
        "Rispetta rigorosamente lo schema JSON richiesto."
    )

def build_user_instruction(level_hint: str) -> str:
    # Compact schema with constraints to shrink output tokens
    return (
        "Fornisci SOLO JSON compatto con questa struttura e nulla di più:\n"
        "{\n"
        '  "c": "<testo_corretto_breve>",\n'
        '  "i": [ {"s":"<span>", "f":"<correzione>", "e":"<spiegazione ≤15 parole>"} ]\n'
        "}\n"
        "Regole:\n"
        "- Max 4 elementi in i.\n"
        "- Spiegazioni ≤ 15 parole.\n"
        "- Non includere testo ripetuto.\n"
        "- Non usare markdown, né backtick.\n"
        "- Non tradurre.\n"
        "- Se nessun errore: i = [].\n"
        f"- Livello: {level_hint}\n"
    )

def feedback_claude_sonnet(
    client: Anthropic,
    model: str,
    text_it: str,
    level_hint: str = RUBRIC_HINT,
    max_output_tokens: int = 500
) -> Tuple[Dict, Dict]:
    """
    Call Claude Sonnet with compact prompt + schema.
    Returns:
        parsed (dict with keys "c", "i"),
        usage (dict with input_tokens, output_tokens, cost_usd)
    """
    system_msg = build_compact_prompt(level_hint)
    user_schema = build_user_instruction(level_hint)

    # Compose a single user message with schema + text to reduce overhead
    user_content = (
        user_schema
        + "\n\nTESTO DA ANALIZZARE (italiano):\n"
        + text_it
    )

    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        temperature=0,
        system=system_msg,
        messages=[{"role": "user", "content": user_content}]
    )

    txt = ""
    try:
        # SDK returns a list of content blocks; first is text
        txt = (resp.content[0].text if resp.content else "").strip()
    except Exception:
        txt = ""

    parsed = {"c": "", "i": []}
    try:
        # Some models may prepend stray characters; find first '{'
        start = txt.find("{")
        if start > 0:
            txt = txt[start:]
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            parsed = {"c": "", "i": []}
        # ensure keys exist
        parsed.setdefault("c", "")
        parsed.setdefault("i", [])
        # enforce caps locally
        if isinstance(parsed["i"], list) and len(parsed["i"]) > 4:
            parsed["i"] = parsed["i"][:4]
    except Exception:
        parsed = {"c": "", "i": []}

    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    # Pricing: Sonnet ≈ $3/M input, $15/M output
    cost = in_tok * 3e-6 + out_tok * 15e-6

    usage = {"input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost}
    return parsed, usage

# ---------------------------
# ------------ UI -----------
# ---------------------------

st.set_page_config(
    page_title="Italian Speaking Practice — Claude Sonnet",
    page_icon="🇮🇹",
    layout="wide"
)

st.title("🇮🇹 Italian Speaking Practice — Claude Sonnet (Optimized)")
warning_banner()

with st.sidebar:
    st.header("Settings")

    # Claude API key
    api_key = get_api_key()
    if not api_key:
        st.warning("Add **ANTHROPIC_API_KEY** to Streamlit Secrets or environment variables.")

    # Claude model (allow override)
    model_name = st.text_input(
        "Claude model",
        value=DEFAULT_SONNET_MODEL,
        help="Default Sonnet model. Change if your tenant exposes a newer snapshot."
    )

    # ASR settings
    st.subheader("ASR (Whisper)")
    whisper_model = st.selectbox("Model size", ["small", "base", "tiny"], index=0)
    compute_type = st.selectbox("Compute type", ["int8_float16", "int8", "float16"], index=0)

    # Optimization toggles
    st.subheader("Token optimization")
    compact_mode = st.checkbox("Compact mode (recommended)", True,
                               help="Short schema, ≤4 issues, brief explanations")
    remove_fillers = st.checkbox("Remove obvious fillers before analysis", True,
                                 help="Trims 'ehm', 'cioè', 'tipo', etc. (text only)")
    max_words_cap = st.slider("Max words sent to Claude", 100, 300, 180, 10)

    # TTS
    auto_tts = st.checkbox("Play corrected version (TTS)", False)

col1, col2 = st.columns([1.2, 0.8])

with col2:
    st.subheader("🎯 Prompt")
    if st.button("Get a prompt"):
        st.session_state["prompt"] = PROMPTS[int(time.time()) % len(PROMPTS)]
    st.info(st.session_state.get("prompt", "Click to get a prompt!"))

    st.markdown("**Guideline:** " + RUBRIC_HINT)

with col1:
    st.subheader("🎙️ Record or upload audio (max 60s)")

    wav_bytes: Optional[bytes] = None
    audio_path = None
    input_duration = 0.0

    # --- Microphone (WAV) ---
    MIC_AVAILABLE = False
    try:
        from streamlit_mic_recorder import mic_recorder
        MIC_AVAILABLE = True
    except Exception:
        MIC_AVAILABLE = False

    if MIC_AVAILABLE:
        rec = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            format="wav"
        )
        if rec is not None:
            if isinstance(rec, dict) and rec.get("bytes"):
                wav_bytes = rec["bytes"]
            elif isinstance(rec, (bytes, bytearray)):
                wav_bytes = bytes(rec)

    # --- Uploader (multi-format) ---
    accepted_types = ["wav", "mp3", "m4a", "ogg", "webm"]
    uploaded = st.file_uploader("…or upload an audio file", type=accepted_types)

    # Handle recording (always WAV)
    if wav_bytes:
        audio_path = save_bytes_to_file(wav_bytes, ".wav")
        input_duration = wav_duration_seconds(audio_path)
        if input_duration > MAX_RECORD_SECONDS:
            st.error(f"Recording is {input_duration:.1f}s (> {MAX_RECORD_SECONDS:.0f}s). "
                     "Please record again within the 60s limit.")
            audio_path = None
            input_duration = 0.0
        else:
            st.success(f"Recording captured ({input_duration:.1f}s).")
            st.audio(wav_bytes, format="audio/wav")

    # Handle upload (keep original suffix). We’ll analyze only first 60s.
    elif uploaded:
        file_bytes = uploaded.read()
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
        if suffix.lstrip(".") not in accepted_types:
            st.error("Unsupported file type. Please upload WAV, MP3, M4A, OGG, or WEBM.")
            st.stop()
        audio_path = save_bytes_to_file(file_bytes, suffix)
        if suffix == ".wav":
            input_duration = wav_duration_seconds(audio_path)
            st.success(f"Audio uploaded ({input_duration:.1f}s). We'll analyze the first 60s.")
        else:
            st.success(f"Audio uploaded ({suffix[1:].upper()}). We'll analyze the first 60s.")
        st.audio(file_bytes)

    # ---- Analyze ----
    if st.button("🧠 Analyze") and audio_path:
        # Check API key and client
        if Anthropic is None:
            st.error("The 'anthropic' package is not installed. Please add it to requirements.txt.")
            st.stop()
        if not api_key:
            st.error("ANTHROPIC_API_KEY is missing. Add it to Streamlit Secrets or environment variables.")
            st.stop()

        # Load ASR model
        with st.status("Loading ASR model…", expanded=False) as status:
            try:
                asr_model = load_fw_model(whisper_model, compute_type)
                status.update(label=f"Model ready: {whisper_model}/{compute_type}", state="complete", expanded=False)
            except Exception as e:
                st.error(f"ASR loading error: {e}")
                st.stop()

        # Transcribe
        with st.spinner("Transcribing…"):
            try:
                raw_text, words, seg_stats = transcribe(asr_model, audio_path)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()

        # If this is an upload, analyze only first 60 seconds (truncate words)
        if uploaded and words:
            words_60 = truncate_words_to_seconds(words, TRUNCATE_UPLOAD_SECONDS)
            if words_60:
                text_60 = words_to_text(words_60)
            else:
                text_60 = raw_text
        else:
            text_60 = raw_text

        # Normalize & compact
        compact_text = compact_for_llm(
            text_60,
            remove_fillers=remove_fillers,
            max_words=max_words_cap
        ) if compact_mode else normalize_italian_text(text_60)

        st.markdown("### 📝 Transcription (compacted for analysis)")
        st.write(compact_text if compact_text else "*(no transcription)*")
        if not compact_text.strip():
            st.warning("No text to analyze.")
            st.stop()

        # Low-confidence highlight (informational)
        if words:
            def render_word_spans(words_list):
                spans = []
                for w in words_list:
                    dur = (w["end"] - w["start"]) if w["end"] > w["start"] else 0.0
                    low_conf = (w["conf"] < 0.30) or (dur < 0.06)
                    if low_conf and w["end"] <= (TRUNCATE_UPLOAD_SECONDS if uploaded else MAX_RECORD_SECONDS):
                        spans.append(
                            f"<span style='background:#fff3cd;padding:2px 4px;border-radius:4px'>{w['word']}</span>"
                        )
                    else:
                        spans.append(w["word"])
                return " ".join(spans)

            st.markdown("#### 🔎 Low‑confidence words (highlighted)")
            st.markdown(render_word_spans(words), unsafe_allow_html=True)

        # Claude Sonnet call
        client = get_anthropic_client(api_key)
        with st.spinner("Generating feedback…"):
            try:
                parsed, usage = feedback_claude_sonnet(
                    client=client,
                    model=model_name,
                    text_it=compact_text,
                    level_hint=RUBRIC_HINT,
                    max_output_tokens=400 if compact_mode else 700
                )
            except Exception as e:
                st.error(f"Claude error: {e}")
                st.stop()

        # Render feedback
        st.markdown("### ✍️ Issues & suggestions (Claude Sonnet)")
        issues = parsed.get("i", []) if isinstance(parsed, dict) else []
        if not issues:
            st.info("Nessun errore significativo trovato.")
        else:
            for idx, it in enumerate(issues, 1):
                s = it.get("s", "")
                f = it.get("f", "")
                e = it.get("e", "")
                with st.expander(f"#{idx} {e or 'Suggerimento'}"):
                    st.write(f"**Span:** `{s}`")
                    st.write(f"**Suggerimento:** `{f}`")
                    if e:
                        st.caption(e)

        corrected = parsed.get("c", "").strip() if isinstance(parsed, dict) else ""
        st.markdown("### ✅ Corrected version")
        st.write(corrected if corrected else "(—)")

        if auto_tts and corrected:
            try:
                buf = io.BytesIO()
                gTTS(corrected, lang="it").write_to_fp(buf)
                buf.seek(0)
                st.audio(buf, format="audio/mp3")
                st.download_button(
                    "⬇️ Download audio (MP3)",
                    data=buf,
                    file_name="correction.mp3",
                    mime="audio/mpeg"
                )
            except Exception as e:
                st.warning(f"TTS unavailable: {e}")

        # Indicators & cost
        n_words = len([w for w in compact_text.split() if w.strip()])
        duration_sec = (
            seg_stats[-1]["end"] if seg_stats else 0.0
        )
        wpm = round(n_words / (max(duration_sec, 1e-6) / 60), 1) if duration_sec else 0.0

        st.markdown("### 💰 Usage & indicators")
        cA, cB, cC = st.columns(3)
        cA.metric("Input tokens", usage.get("input_tokens", 0))
        cA.metric("Output tokens", usage.get("output_tokens", 0))
        cB.metric("Estimated cost ($)", round(usage.get("cost_usd", 0.0), 5))
        cB.metric("Words sent", n_words)
        cC.metric("Approx. WPM (transcribed)", wpm)
        cC.metric("Max analyzed seconds",
                  int(TRUNCATE_UPLOAD_SECONDS if uploaded else MAX_RECORD_SECONDS))

st.markdown("---")
st.caption(
    "Feedback powered by Claude Sonnet (API). Pricing reference: ~$3/M input tokens & $15/M output tokens. "
    "Only the first 60s of uploaded audio are analyzed; recordings >60s are rejected."
)
