# app.py — Streamlit Cloud–friendly Italian speaking app (English UI)
# - Uses streamlit-mic-recorder (works on Streamlit Cloud)
# - faster-whisper (tiny, int8)
# - LanguageTool grammar/style
# - WAV-only for simplicity
# - Lightweight metrics

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
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import requests
from gtts import gTTS

# ---------------------------
# ---------- Config ----------
# ---------------------------

DEFAULT_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "https://api.languagetool.org/v2/check")
ASR_MODEL_NAME = "tiny"   # force tiny for Streamlit Cloud
ASR_COMPUTE = "int8"      # fastest CPU compute type

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

@st.cache_resource(show_spinner=False)
def load_fw_model() -> WhisperModel:
    """Load faster-whisper (cached)."""
    return WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type=ASR_COMPUTE)

def save_wav_bytes_to_file(wav_bytes: bytes) -> Tuple[str, float]:
    """Save WAV bytes to a temp file and return path + duration (seconds)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmp.name, "wb") as f:
        f.write(wav_bytes)
    return tmp.name, get_wav_duration(tmp.name)

def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 16000
        return frames / float(rate)

def transcribe(model: WhisperModel, wav_path: str) -> str:
    segments, _ = model.transcribe(
        wav_path,
        language="it",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=1
    )
    return " ".join(s.text.strip() for s in segments if s.text).strip()

def call_languagetool(text: str, endpoint: str) -> Dict:
    r = requests.post(endpoint, data={"text": text, "language": "it"}, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_lt_matches(data: Dict) -> List[LTMatch]:
    out: List[LTMatch] = []
    for m in data.get("matches", []):
        reps = [r.get("value") for r in m.get("replacements", [])]
        rule = m.get("rule") or {}
        out.append(
            LTMatch(
                message=m.get("message", ""),
                offset=m.get("offset", 0),
                length=m.get("length", 0),
                replacements=reps,
                rule_id=rule.get("id", ""),
                sentence=m.get("sentence", ""),
                category=(rule.get("category") or {}).get("id", "")
            )
        )
    return out

def apply_corrections(text: str, matches: List[LTMatch]) -> str:
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
    return ("C1" if score >= 5 else
            "B2" if score >= 3 else
            "B1" if score >= 2 else
            "A2 or lower")

def tts_bytes(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    gTTS(text, lang="it").write_to_fp(buf)
    buf.seek(0)
    return buf

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
        help="Defaults to the public API (rate limited). Set a custom endpoint in Streamlit Secrets for classes."
    )
    auto_tts = st.checkbox("Play the corrected version (TTS)", False)
    st.caption("This build forces the tiny model for stable performance on Streamlit Cloud.")

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

    # ---- Microphone recorder ----
    # mic_recorder returns raw WAV bytes by default (when as_raw=True in latest versions),
    # but many versions return a NumPy array or bytes-like. We handle both.
    rec = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=True,
        format="wav"  # ensure WAV for simplicity on Cloud
    )

    wav_bytes: Optional[bytes] = None

    if rec is not None:
        # Some versions return a dict with "bytes", others return raw bytes/array.
        if isinstance(rec, dict) and "bytes" in rec and rec["bytes"]:
            wav_bytes = rec["bytes"]
        elif isinstance(rec, (bytes, bytearray)):
            wav_bytes = bytes(rec)
        elif hasattr(rec, "tobytes"):  # e.g., NumPy array
            wav_bytes = rec.tobytes()

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

    if st.button("🧠 Analyze") and audio_path:
        with st.status("Loading ASR model…"):
            model = load_fw_model()

        with st.spinner("Transcribing…"):
            text = transcribe(model, audio_path)

        st.markdown("### 📝 Transcription")
        st.write(text if text else "*(no transcription)*")
        if not text.strip():
            st.warning("No text to analyze.")
            st.stop()

        with st.spinner("Analyzing grammar…"):
            try:
                lt_json = call_languagetool(text, lt_endpoint)
                matches = parse_lt_matches(lt_json)
            except Exception as e:
                st.error(f"LanguageTool error: {e}")
                matches = []

        st.markdown("### ✍️ Issues & suggestions")
        if not matches:
            st.info("No significant issues found (or API limit reached).")
        else:
            for i, m in enumerate(matches, 1):
                suggestion = m.replacements[0] if m.replacements else "—"
                with st.expander(f"#{i} {m.message}"):
                    st.write(f"**Sentence:** {m.sentence}")
                    st.write(f"**Suggestion:** `{suggestion}`")
                    st.write(f"Rule: `{m.rule_id}` • Category: `{m.category}`")

        corrected = apply_corrections(text, matches)
        st.markdown("### ✅ Corrected version (auto-generated)")
        st.write(corrected)

        if auto_tts and corrected.strip():
            audio = tts_bytes(corrected)
            st.audio(audio, format="audio/mp3")
            st.download_button("⬇️ Download audio (MP3)", data=audio, file_name="correction.mp3", mime="audio/mpeg")

        metrics = compute_text_metrics(text)
        wpm = round(metrics["n_words"] / (duration / 60), 1) if duration > 0 else 0

        st.markdown("### 📊 Indicators")
        cA, cB, cC = st.columns(3)
        cA.metric("Words", metrics["n_words"])
        cA.metric("Sentences", metrics["n_sentences"])
        cB.metric("TTR", metrics["ttr"])
        cB.metric("Avg. sentence length", metrics["avg_sentence_len"])
        cC.metric("WPM", wpm)
        cC.metric("Level (est.)", estimate_cefr(metrics))
