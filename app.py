# ---------------------------------------------------------
# Italian Speaking Practice (Updated Edition)
# - ASR model selector (tiny/base/small)
# - Word-level timestamps + low-confidence highlighting
# - Italian apostrophe normalization
# - Improved transcription with temperature fallback
# - Uploader now accepts WAV/MP3/M4A/OGG/WEBM
# - Preserves LT + TTS features (free tools only)
# ---------------------------------------------------------

import os
import io
import time
import json
import wave
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import streamlit as st

# --- Microphone component (safe import with fallback) ---
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

from faster_whisper import WhisperModel
import requests
from gtts import gTTS
import re

# ---------------------------
# ---------- Config ----------
# ---------------------------

DEFAULT_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "https://api.languagetool.org/v2/check")

PROMPTS = [
    "Describe your ideal day from start to finish.",
    "Tell us about a trip that changed your life.",
    "Explain a typical dish from your region and how to prepare it.",
    "Discuss a social issue in Italy and propose some solutions.",
    "Compare life in the city and the countryside: pros and cons.",
    "If you could change one law, which would it be and why?",
    "Talk about a book or film that impacted you and explain why."
]

RUBRIC = [
    "✔ Clarity & coherence: ideas are organized and connected.",
    "✔ Vocabulary: variety and natural collocations.",
    "✔ Grammar: agreement and verb tenses are appropriate.",
    "✔ Pronunciation & fluency: steady pace, natural pauses.",
    "✔ Register & style: appropriate to the context."
]

# ---------------------------
# ---------- Data -----------
# ---------------------------

@dataclass
class LTMatch:
    message: str
    offset: int
    length: int
    replacements: List[str]
    rule_id: str
    sentence: str
    category: str

# ---------------------------
# -------- Utilities --------
# ---------------------------

def screenshot_safe(s: str) -> str:
    """Avoid code formatting issues if suggestions contain backticks."""
    return s.replace("`", "´")

@st.cache_resource(show_spinner=False)
def load_fw_model(name: str, compute: str) -> WhisperModel:
    """Load faster-whisper (cached) with chosen model + compute type."""
    return WhisperModel(name, device="cpu", compute_type=compute)

def save_bytes_to_file(data: bytes, suffix: str) -> str:
    """Save raw bytes to a temp file with provided suffix (e.g., '.mp3', '.wav')."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        f.write(data)
    return tmp.name

def wav_duration_seconds(path: str) -> float:
    """Duration from WAV header; for non-WAV we compute duration from transcription later."""
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            return frames / float(rate)
    except wave.Error:
        return 0.0

# ---------------------------
# ---- Italian normalization -
# ---------------------------

MULTISPACE = re.compile(r"\s{2,}")

def normalize_italian_text(text: str) -> str:
    """Fix spacing around apostrophes and common elisions."""
    # Merge "l ' amico" → "l'amico"
    text = re.sub(
        r"\b([lLdDaAeEoOuU]l?)\s*'\s*([a-zA-Zàèéìòóù])",
        r"\1'\2",
        text
    )
    # Clean stray spaces
    text = text.replace(" ’ ", "’").replace(" ' ", "'")
    text = text.replace(" ’", "’").replace("’ ", "’")
    text = text.replace(" '", "'").replace("' ", "'")
    text = MULTISPACE.sub(" ", text)
    return text.strip()

# ---------------------------
# ---- Transcription w/ timestamps
# ---------------------------

def transcribe(model: WhisperModel, audio_path: str):
    """
    Transcribe Italian audio with word timestamps + temperature fallback.
    Returns (full_text, words, seg_stats).
    """
    temperature = [0.0, 0.2, 0.4]

    segments, _info = model.transcribe(
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
        initial_prompt=(
            "Trascrivi in italiano standard senza riformulare. "
            "Evita correzioni non necessarie. Non tradurre."
        ),
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
                conf = max(min((avg + 5) / 5, 1.0), 0.0)  # heuristic confidence proxy
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

# ---------------------------
# ---- LanguageTool API -----
# ---------------------------

def call_languagetool(text: str, endpoint: str) -> Dict:
    payload = {
        "text": text,
        "language": "it",
        "disabledCategories": "TYPOS,CASING,PUNCTUATION",
    }
    headers = {
        "User-Agent": "KU-Italian-Speaking-Coach/1.0 (+educational use)",
    }
    try:
        r = requests.post(endpoint, data=payload, headers=headers, timeout=15)
        if r.status_code == 429:
            return {"ok": False, "rate_limited": True, "data": {}}
        r.raise_for_status()
        return {"ok": True, "rate_limited": False, "data": r.json()}
    except requests.exceptions.Timeout:
        return {"ok": False, "rate_limited": False, "data": {}}
    except requests.RequestException:
        return {"ok": False, "rate_limited": False, "data": {}}

@dataclass
class LTMatch:
    message: str
    offset: int
    length: int
    replacements: List[str]
    rule_id: str
    sentence: str
    category: str

def parse_lt_matches(data: Dict) -> List[LTMatch]:
    out = []
    for m in data.get("matches", []):
        rule = m.get("rule") or {}
        cat = (rule.get("category") or {}).get("id", "")
        if cat in {"TYPOS", "CASING", "PUNCTUATION"}:
            continue
        reps = [r.get("value") for r in m.get("replacements", [])]
        out.append(
            LTMatch(
                message=m.get("message", ""),
                offset=m.get("offset", 0),
                length=m.get("length", 0),
                replacements=reps,
                rule_id=rule.get("id", ""),
                sentence=m.get("sentence", ""),
                category=cat
            )
        )
    return out

def apply_corrections(text: str, matches: List[LTMatch]) -> str:
    """Apply first replacement for each match (right-to-left)."""
    out = text
    for m in sorted(matches, key=lambda x: x.offset, reverse=True):
        if m.replacements and m.length > 0:
            s, e = m.offset, m.offset + m.length
            if 0 <= s <= len(out) and 0 <= e <= len(out):
                out = out[:s] + m.replacements[0] + out[e:]
    return out

def compute_text_metrics(text: str) -> Dict[str, float]:
    words = [w.strip(".,;:!?\"'()[]{}").lower() for w in text.split() if w.strip()]
    if not words:
        return {"n_words": 0, "n_sentences": 0, "ttr": 0.0, "avg_sentence_len": 0.0}
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    return {
        "n_words": len(words),
        "n_sentences": len(sentences),
        "ttr": round(len(set(words)) / len(words), 3),
        "avg_sentence_len": round(len(words) / (len(sentences) or 1), 2)
    }

def estimate_cefr(metrics: Dict[str, float]) -> str:
    score = 0
    if metrics["ttr"] >= 0.6: score += 2
    elif metrics["ttr"] >= 0.45: score += 1
    if metrics["avg_sentence_len"] >= 18: score += 2
    elif metrics["avg_sentence_len"] >= 12: score += 1
    return (
        "C1 (approx.)" if score >= 5 else
        "B2 (approx.)" if score >= 3 else
        "B1 (approx.)" if score >= 2 else
        "A2 or lower (approx.)"
    )

def tts_bytes(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    gTTS(text, lang="it").write_to_fp(buf)
    buf.seek(0)
    return buf

# ---------------------------
# ---- Session debounce -----
# ---------------------------

if "last_lt_call_ts" not in st.session_state:
    st.session_state["last_lt_call_ts"] = 0.0

LT_MIN_INTERVAL_SEC = 2.0

# ---------------------------
# ------------ UI -----------
# ---------------------------

st.set_page_config(
    page_title="Italian Speaking Practice — Updated",
    page_icon="🇮🇹",
    layout="wide"
)

st.title("🇮🇹 Italian Speaking Practice — Updated Edition")

# Sidebar
with st.sidebar:
    st.header("Settings")

    lt_endpoint = st.text_input(
        "LanguageTool endpoint",
        value=DEFAULT_LT_ENDPOINT
    )

    auto_tts = st.checkbox("Play the corrected version (TTS)", False)

    st.subheader("ASR Model")
    model_name = st.selectbox("Model size", ["tiny", "base", "small"], index=0)
    compute_type = st.selectbox("Compute type", ["int8", "int8_float16", "float16"], index=0)

    st.caption("For best accuracy: try 'small' + 'int8_float16' (slower but better).")

col1, col2 = st.columns([1.2, 0.8])

with col2:
    st.subheader("🎯 Prompt & Rubric")

    if st.button("Get a prompt"):
        st.session_state["prompt"] = PROMPTS[int(time.time()) % len(PROMPTS)]

    st.info(st.session_state.get("prompt", "Click the button to get a prompt!"))

    st.write("**Guideline rubric:**")
    for item in RUBRIC:
        st.write(item)

with col1:
    st.subheader("🎙️ Record or upload audio")

    wav_bytes: Optional[bytes] = None

    # Recorder (WAV)
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

    # ---- Uploader: accept multiple types ----
    accepted_types = ["wav", "mp3", "m4a", "ogg", "webm"]
    uploaded = st.file_uploader("…or upload an audio file", type=accepted_types)

    audio_path = None
    input_duration = 0.0  # only reliable for WAV; for others we'll compute from segments

    # Recorder path (always WAV)
    if wav_bytes:
        audio_path = save_bytes_to_file(wav_bytes, ".wav")
        input_duration = wav_duration_seconds(audio_path)
        st.success(f"Recording captured ({input_duration:.1f}s).")
        st.audio(wav_bytes, format="audio/wav")

    # Uploaded file path (keep original suffix)
    elif uploaded:
        file_bytes = uploaded.read()
        # Determine suffix from filename (fallback to .wav if missing)
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
        # Safety: restrict to accepted suffixes even if the extension is odd
        if suffix.lstrip(".") not in accepted_types:
            st.error("Unsupported file type. Please upload WAV, MP3, M4A, OGG, or WEBM.")
            st.stop()
        audio_path = save_bytes_to_file(file_bytes, suffix)
        # Only compute header-based duration for WAV; others computed after transcription.
        if suffix == ".wav":
            input_duration = wav_duration_seconds(audio_path)
            st.success(f"Audio uploaded ({input_duration:.1f}s).")
        else:
            st.success(f"Audio uploaded ({suffix[1:].upper()}).")
        st.audio(file_bytes)

    # ---- Analyze ----
    if st.button("🧠 Analyze") and audio_path:
        with st.status("Loading ASR model…", expanded=False) as status:
            try:
                model = load_fw_model(model_name, compute_type)
                status.update(
                    label=f"Model ready: {model_name}/{compute_type}",
                    state="complete",
                    expanded=False
                )
            except Exception as e:
                st.error(f"Model loading error: {e}")
                st.stop()

        with st.spinner("Transcribing…"):
            try:
                text, words, seg_stats = transcribe(model, audio_path)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()

        text = normalize_italian_text(text)

        st.markdown("### 📝 Transcription")
        st.write(text if text else "*(no transcription)*")

        # Highlight uncertain words
        if words:
            def render_word_spans(words_list):
                spans = []
                for w in words_list:
                    dur = (w["end"] - w["start"]) if w["end"] > w["start"] else 0.0
                    low_conf = (w["conf"] < 0.30) or (dur < 0.06)
                    if low_conf:
                        spans.append(
                            f"<span style='background:#fff3cd;padding:2px 4px;border-radius:4px'>{w['word']}</span>"
                        )
                    else:
                        spans.append(w["word"])
                return " ".join(spans)

            st.markdown("#### 🔎 Low‑confidence words (highlighted)")
            st.markdown(render_word_spans(words), unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download words+timestamps (JSON)",
                data=json.dumps({"words": words, "segments": seg_stats}, ensure_ascii=False, indent=2),
                file_name="alignment.json",
                mime="application/json",
            )

        # --- Grammar analysis (LanguageTool) ---
        word_count = len([w for w in text.split() if w.strip()])
        should_call_lt = word_count >= 5

        now = time.time()
        if now - st.session_state["last_lt_call_ts"] < LT_MIN_INTERVAL_SEC:
            should_call_lt = False
        else:
            st.session_state["last_lt_call_ts"] = now

        lt_result = {"ok": False, "rate_limited": False, "data": {}}
        matches: List[LTMatch] = []

        with st.spinner("Analyzing grammar…"):
            if should_call_lt:
                lt_result = call_languagetool(text, lt_endpoint)
                if lt_result["ok"]:
                    matches = parse_lt_matches(lt_result["data"])
                else:
                    matches = []

        st.markdown("### ✍️ Issues & suggestions")

        if lt_result.get("rate_limited"):
            st.warning("LanguageTool public API limit reached. Try again soon.")
        elif not should_call_lt:
            if word_count < 5:
                st.info("Speak a bit more to get grammar/style feedback (≥ 5 words).")
            else:
                st.info("Analyzing… please try again in a moment.")
        elif not matches:
            st.info("No significant grammar/style issues found.")
        else:
            for i, m in enumerate(matches, 1):
                suggestion = m.replacements[0] if m.replacements else "—"
                with st.expander(f"#{i} {m.message}"):
                    st.write(f"**Sentence:** {m.sentence}")
                    st.write(f"**Suggestion:** `{screenshot_safe(suggestion)}`")
                    st.write(f"Rule: `{m.rule_id}` • Category: `{m.category}`")

        corrected = normalize_italian_text(apply_corrections(text, matches))

        st.markdown("### ✅ Corrected version")
        st.write(corrected)

        if auto_tts and corrected.strip():
            try:
                audio = tts_bytes(corrected)
                st.audio(audio, format="audio/mp3")
                st.download_button(
                    "⬇️ Download audio (MP3)",
                    data=audio,
                    file_name="correction.mp3",
                    mime="audio/mpeg"
                )
            except Exception as e:
                st.warning(f"TTS unavailable: {e}")

        # Indicators
        metrics = compute_text_metrics(text)

        # Use WAV header duration if available; otherwise derive from last segment end time
        if input_duration and input_duration > 0:
            duration_sec = input_duration
        else:
            duration_sec = seg_stats[-1]["end"] if seg_stats else 0.0

        wpm = round(metrics["n_words"] / (duration_sec / 60), 1) if duration_sec > 0 else 0

        st.markdown("### 📊 Indicators")
        cA, cB, cC = st.columns(3)
        cA.metric("Words", metrics["n_words"])
        cA.metric("Sentences", metrics["n_sentences"])
        cB.metric("TTR", metrics["ttr"])
        cB.metric("Avg. sentence length", metrics["avg_sentence_len"])
        cC.metric("WPM", wpm)
        cC.metric("Level (estimate)", estimate_cefr(metrics))

st.markdown("---")
st.caption(
    "Uploader accepts WAV/MP3/M4A/OGG/WEBM. Duration is from WAV header when available; "
    "otherwise estimated from transcription timing."
)
