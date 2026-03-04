# app.py — Streamlit Cloud–friendly Italian speaking app (English UI)
# - Uses streamlit-mic-recorder (with graceful fallback to upload)
# - faster-whisper (tiny, CPU, int8) tuned for better results on Cloud
# - LanguageTool grammar/style tuned for speech (writing nits suppressed)
# - Robust LT handling: rate-limit detection, timeouts, debounce, short-utterance guard
# - WAV-only to avoid extra media dependencies
# - Lightweight text metrics (WPM, TTR, avg sentence length)
# - Optional TTS (gTTS) to hear the corrected version

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

# ---------------------------
# ---------- Config ----------
# ---------------------------

# Public LanguageTool by default (rate-limited).
# You can override via Streamlit "Secrets" or environment variable.
DEFAULT_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "https://api.languagetool.org/v2/check")

# Cloud-safe defaults
ASR_MODEL_NAME = "tiny"   # Force tiny for Streamlit Cloud
ASR_COMPUTE = "int8"      # Fastest CPU compute type

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
def load_fw_model() -> WhisperModel:
    """Load faster-whisper (cached). Tiny+int8 is Cloud-friendly."""
    return WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type=ASR_COMPUTE)

def save_wav_bytes_to_file(wav_bytes: bytes) -> Tuple[str, float]:
    """Save WAV bytes to a temp file and return path + duration (seconds)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmp.name, "wb") as f:
        f.write(wav_bytes)
    return tmp.name, get_wav_duration(tmp.name)

def get_wav_duration(path: str) -> float:
    """Read WAV header to compute duration."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 16000
        return frames / float(rate)

# ---- Transcribe with improved decoding/VAD for tiny ----
def transcribe(model: WhisperModel, wav_path: str) -> str:
    """
    Transcribe Italian speech using faster-whisper with settings tuned to
    squeeze better accuracy out of the 'tiny' model on learner speech.

    Key tweaks:
      - beam_size=5, best_of=5 for better sequence selection
      - temperature=0.0 for deterministic decoding (fewer hallucinations)
      - condition_on_previous_text=False to avoid error carryover
      - initial_prompt in Italian to bias toward standard Italian output
      - VAD with slightly shorter silence to segment learner speech
    """
    segments, _ = model.transcribe(
        wav_path,
        language="it",
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 600},
        beam_size=5,
        best_of=5,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt=(
            "Trascrivi in italiano standard senza riformulare. "
            "Evita correzioni di maiuscole non necessarie. "
            "Non tradurre, mantieni la lingua italiana."
        ),
        condition_on_previous_text=False
    )
    return " ".join(s.text.strip() for s in segments if s.text).strip()

# ---- LanguageTool call focused on speech grammar/style + robust handling ----
def call_languagetool(text: str, endpoint: str) -> Dict:
    """
    Call LanguageTool and suppress writing-centric categories.
    Returns a dict with keys: {"ok": bool, "rate_limited": bool, "data": dict}
    """
    payload = {
        "text": text,
        "language": "it",
        # Keep feedback speech-oriented:
        "disabledCategories": "TYPOS,CASING,PUNCTUATION",
    }
    headers = {
        "User-Agent": "KU-Italian-Speaking-Coach/1.0 (+educational use)",
    }

    try:
        r = requests.post(endpoint, data=payload, headers=headers, timeout=15)
        # Public API may return 429 for bursts
        if r.status_code == 429:
            return {"ok": False, "rate_limited": True, "data": {}}
        r.raise_for_status()
        return {"ok": True, "rate_limited": False, "data": r.json()}
    except requests.exceptions.Timeout:
        return {"ok": False, "rate_limited": False, "data": {}}
    except requests.RequestException:
        # Network / other server errors
        return {"ok": False, "rate_limited": False, "data": {}}

def parse_lt_matches(data: Dict) -> List[LTMatch]:
    """
    Parse LT matches AND defensively filter out writing-only categories that
    might still slip through (double safety).
    """
    out: List[LTMatch] = []
    for m in data.get("matches", []):
        rule = m.get("rule") or {}
        cat  = (rule.get("category") or {}).get("id", "")
        # Filter writing-only categories at UI layer as well
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
    """Apply first suggested replacement for each match (right-to-left)."""
    out = text
    for m in sorted(matches, key=lambda x: x.offset, reverse=True):
        if m.replacements and m.length > 0:
            s, e = m.offset, m.offset + m.length
            if 0 <= s <= len(out) and 0 <= e <= len(out):
                out = out[:s] + m.replacements[0] + out[e:]
    return out

def compute_text_metrics(text: str) -> Dict[str, float]:
    """Lightweight text metrics: words, sentences, TTR, avg sentence length."""
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
    """Very rough CEFR-ish hint based on lexical diversity and sentence length."""
    score = 0
    if metrics["ttr"] >= 0.6: score += 2
    elif metrics["ttr"] >= 0.45: score += 1
    if metrics["avg_sentence_len"] >= 18: score += 2
    elif metrics["avg_sentence_len"] >= 12: score += 1
    return ("C1 (approx.)" if score >= 5 else
            "B2 (approx.)" if score >= 3 else
            "B1 (approx.)" if score >= 2 else
            "A2 or lower (approx.)")

def tts_bytes(text: str) -> io.BytesIO:
    """Generate MP3 TTS audio for the corrected text (Italian)."""
    buf = io.BytesIO()
    gTTS(text, lang="it").write_to_fp(buf)
    buf.seek(0)
    return buf

# ---------------------------
# ---- Session-level UX -----
# ---------------------------

# Simple per-session debounce to avoid spamming LT when users click rapidly
if "last_lt_call_ts" not in st.session_state:
    st.session_state["last_lt_call_ts"] = 0.0

LT_MIN_INTERVAL_SEC = 2.0  # don't call LT more frequently than every 2s

# ---------------------------
# ------------ UI -----------
# ---------------------------

st.set_page_config(page_title="Italian Speaking Practice (Cloud)", page_icon="🇮🇹", layout="wide")
st.title("🇮🇹 Italian Speaking Practice — Cloud Edition (tiny ASR model)")

with st.sidebar:
    st.header("Settings")
    lt_endpoint = st.text_input(
        "LanguageTool endpoint",
        value=DEFAULT_LT_ENDPOINT,
        help="Defaults to the public API (rate limited). Use Streamlit Secrets to set a custom endpoint for classes."
    )
    auto_tts = st.checkbox("Play the corrected version (TTS)", False)
    st.caption(
        "Using the public LanguageTool endpoint may hit rate limits during busy periods. "
        "For classes, consider a private LT server and set its URL here."
    )
    st.caption("This build forces the tiny ASR model for stability on Streamlit Cloud. WAV input only.")

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
    st.subheader("🎙️ Record or upload a WAV")

    # ---- Microphone recorder (with graceful fallback) ----
    wav_bytes: Optional[bytes] = None

    if MIC_AVAILABLE:
        rec = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            format="wav"  # ensure WAV output for simplicity
        )
        if rec is not None:
            # Normalize different return formats across versions
            if isinstance(rec, dict) and rec.get("bytes"):
                wav_bytes = rec["bytes"]
            elif isinstance(rec, (bytes, bytearray)):
                wav_bytes = bytes(rec)
            elif hasattr(rec, "tobytes"):  # e.g., NumPy array
                wav_bytes = rec.tobytes()
    else:
        st.info("Microphone recorder is unavailable here. Please upload a WAV instead.")

    uploaded = st.file_uploader("…or upload a WAV file", type=["wav"])

    audio_path = None
    duration = 0.0

    if wav_bytes:
        audio_path, duration = save_wav_bytes_to_file(wav_bytes)
        st.success(f"Recording captured ({duration:.1f}s).")
        st.audio(wav_bytes, format="audio/wav")
    elif uploaded:
        file_bytes = uploaded.read()
        audio_path, duration = save_wav_bytes_to_file(file_bytes)
        st.success(f"WAV uploaded ({duration:.1f}s).")
        st.audio(file_bytes, format="audio/wav")

    # ---- Analyze button ----
    if st.button("🧠 Analyze") and audio_path:
        # Load ASR model (cached)
        with st.status("Loading ASR model…", expanded=False) as status:
            try:
                model = load_fw_model()
                status.update(label="Model ready", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Model loading error: {e}")
                st.stop()

        # Transcribe
        with st.spinner("Transcribing…"):
            try:
                text = transcribe(model, audio_path)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()

        st.markdown("### 📝 Transcription")
        st.write(text if text else "*(no transcription)*")
        if not text.strip():
            st.warning("No text to analyze.")
            st.stop()

        # --- Decide whether to call LT now (short utterance + debounce) ---
        min_words_for_lt = 5
        word_count = len([w for w in text.split() if w.strip()])
        should_call_lt = word_count >= min_words_for_lt

        now = time.time()
        if now - st.session_state["last_lt_call_ts"] < LT_MIN_INTERVAL_SEC:
            should_call_lt = False  # too soon; skip this round silently
        else:
            st.session_state["last_lt_call_ts"] = now

        # Grammar & style (speech-focused) with robust handling
        lt_result = {"ok": False, "rate_limited": False, "data": {}}
        matches: List[LTMatch] = []
        with st.spinner("Analyzing grammar…"):
            if should_call_lt:
                lt_result = call_languagetool(text, lt_endpoint)
                if lt_result["ok"]:
                    matches = parse_lt_matches(lt_result["data"])
            else:
                matches = []

        # Clear messaging for users
        st.markdown("### ✍️ Issues & suggestions (speech-focused)")
        if lt_result.get("rate_limited"):
            st.warning("LanguageTool public API limit reached. Please wait a few seconds and try again, or configure a private endpoint in Settings.")
        elif not should_call_lt:
            if word_count < min_words_for_lt:
                st.info("Speak a bit more to get grammar/style feedback (≥ 5 words).")
            else:
                st.info("Analyzing… please try again in a second.")
        elif not matches:
            st.info("No significant grammar/style issues found.")
        else:
            for i, m in enumerate(matches, 1):
                suggestion = m.replacements[0] if m.replacements else "—"
                with st.expander(f"#{i} {m.message}"):
                    st.write(f"**Sentence:** {m.sentence}")
                    st.write(f"**Suggestion:** `{screenshot_safe(suggestion)}`")
                    st.write(f"Rule: `{m.rule_id}` • Category: `{m.category}`")

        # Corrected (grammar-focused, not capitalizing for 'writing')
        corrected = apply_corrections(text, matches)
        st.markdown("### ✅ Corrected version (grammar-focused)")
        st.write(corrected)

        # TTS
        if auto_tts and corrected.strip():
            try:
                audio = tts_bytes(corrected)
                st.audio(audio, format="audio/mp3")
                st.download_button("⬇️ Download audio (MP3)", data=audio, file_name="correction.mp3", mime="audio/mpeg")
            except Exception as e:
                st.warning(f"TTS unavailable: {e}")

        # Metrics
        metrics = compute_text_metrics(text)
        wpm = round(metrics["n_words"] / (duration / 60), 1) if duration > 0 else 0

        st.markdown("### 📊 Indicators")
        cA, cB, cC = st.columns(3)
        cA.metric("Words", metrics["n_words"])
        cA.metric("Sentences", metrics["n_sentences"])
        cB.metric("TTR (lexical diversity)", metrics["ttr"])
        cB.metric("Avg. sentence length", metrics["avg_sentence_len"])
        cC.metric("WPM", wpm)
        cC.metric("Level (estimate)", estimate_cefr(metrics))

st.markdown("---")
st.caption(
    "This build is optimized for Streamlit Cloud: it forces the tiny ASR model and accepts WAV input only. "
    "For heavier classroom use, consider setting a self-hosted LanguageTool endpoint via Secrets."
)
