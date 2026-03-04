# app.py — Streamlit Cloud–friendly Italian speaking app (English UI)
# - Mic recorder via audio_recorder_streamlit (works on Streamlit Cloud)
# - faster-whisper (tiny, int8) forced for speed & reliability
# - LanguageTool grammar/style via public API (or custom endpoint)
# - Lightweight metrics (no librosa): WPM + basic text stats
# - Optional TTS (gTTS)

import os
import io
import time
import json
import wave
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import requests
from gtts import gTTS

# ---------------------------
# ---------- Config ----------
# ---------------------------

# Public LanguageTool by default (rate-limited). You can override via Streamlit "Secrets" or env.
DEFAULT_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "https://api.languagetool.org/v2/check")

ASR_MODEL_NAME = "tiny"      # Force tiny on Streamlit Cloud
ASR_COMPUTE = "int8"         # Fastest CPU compute type

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
    "✔ Register & style: appropriate to the context (formal/informal)."
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
    """Load faster-whisper model (cached). On Cloud we stay on CPU with int8."""
    return WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type=ASR_COMPUTE)

def save_wav_bytes_to_file(wav_bytes: bytes) -> Tuple[str, float]:
    """Save WAV bytes to a temp file and return path + duration (seconds)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmp.name, "wb") as f:
        f.write(wav_bytes)
    duration_sec = get_wav_duration(tmp.name)
    return tmp.name, duration_sec

def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0

def transcribe(model: WhisperModel, wav_path: str) -> str:
    segments, _ = model.transcribe(
        wav_path,
        language="it",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
        beam_size=1
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text).strip()

def call_languagetool(text: str, endpoint: str) -> Dict:
    payload = {"text": text, "language": "it"}
    r = requests.post(endpoint, data=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_lt_matches(raw_json: Dict) -> List[LTMatch]:
    out: List[LTMatch] = []
    for m in raw_json.get("matches", []):
        replacements = [rep.get("value") for rep in m.get("replacements", [])]
        rule = m.get("rule") or {}
        out.append(
            LTMatch(
                message=m.get("message", ""),
                offset=m.get("offset", 0),
                length=m.get("length", 0),
                replacements=replacements,
                rule_id=rule.get("id", ""),
                sentence=m.get("sentence", ""),
                category=(rule.get("category") or {}).get("id", "")
            )
        )
    return out

def apply_corrections(text: str, matches: List[LTMatch]) -> str:
    """Apply single best replacement for each match (right-to-left to preserve offsets)."""
    corrected = text
    for m in sorted(matches, key=lambda x: x.offset, reverse=True):
        if m.replacements and m.length > 0:
            s, e = m.offset, m.offset + m.length
            if 0 <= s <= len(corrected) and 0 <= e <= len(corrected):
                corrected = corrected[:s] + m.replacements[0] + corrected[e:]
    return corrected

def compute_text_metrics(text: str) -> Dict[str, float]:
    tokens = [t for t in text.replace("\n", " ").split(" ") if t.strip()]
    words = [t.strip(".,;:!?\"'()[]{}").lower() for t in tokens if t.strip()]
    if not words:
        return {"n_words": 0, "n_sentences": 0, "ttr": 0.0, "avg_sentence_len": 0.0}
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    avg_sentence_len = len(words) / (len(sentences) if sentences else 1)
    return {
        "n_words": len(words),
        "n_sentences": len(sentences),
        "ttr": round(ttr, 3),
        "avg_sentence_len": round(avg_sentence_len, 2),
    }

def estimate_cefr(metrics: Dict[str, float]) -> str:
    ttr = metrics.get("ttr", 0.0)
    asl = metrics.get("avg_sentence_len", 0.0)
    score = 0
    if ttr >= 0.6: score += 2
    elif ttr >= 0.45: score += 1
    if asl >= 18: score += 2
    elif asl >= 12: score += 1
    if score >= 5: return "C1 (approx.)"
    if score >= 3: return "B2 (approx.)"
    if score >= 2: return "B1 (approx.)"
    return "A2 or lower (approx.)"

def tts_bytes_io(text: str, lang: str = "it") -> io.BytesIO:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf

# ---------------------------
# ------------ UI -----------
# ---------------------------

st.set_page_config(page_title="Italian Speaking Practice (Cloud)", page_icon="🇮🇹", layout="wide")
st.title("🇮🇹 Italian Speaking Practice — Cloud Edition (model: tiny)")

with st.sidebar:
    st.header("⚙️ Settings")
    lt_endpoint = st.text_input(
        "LanguageTool endpoint",
        value=DEFAULT_LT_ENDPOINT,
        help="Defaults to the public service (rate limited). You can set a custom endpoint via Streamlit Secrets."
    )
    auto_tts = st.checkbox("Play the corrected version (TTS)", value=False)
    st.markdown("---")
    st.caption("This build is optimized for Streamlit Cloud: it forces the ASR model **tiny** and accepts WAV input.")

# Prompt & rubric
c1, c2 = st.columns([1.2, 0.8])
with c2:
    st.subheader("🎯 Prompt & Rubric")
    if st.button("Give me a prompt"):
        st.session_state["prompt"] = PROMPTS[int(time.time()) % len(PROMPTS)]
    st.info(st.session_state.get("prompt", "Click the button to get a prompt!"))
    st.write("**Evaluation rubric (for guidance):**")
    for item in RUBRIC:
        st.write(item)

with c1:
    st.subheader("🎙️ Record or upload a WAV")
    st.caption("Tip: record 30–60 seconds in a quiet environment.")
    # Mic recorder (returns WAV bytes). Works on Streamlit Cloud.
    wav_bytes = audio_recorder(
        pause_threshold=2.0,  # seconds of silence to auto-stop
        sample_rate=16000,    # 16 kHz mono
        text="Click to record",
        recording_color="#e06c75",
        neutral_color="#98c379",
        icon_name="microphone",
        icon_size="2x",
    )

    uploaded_wav = st.file_uploader("…or upload a WAV file", type=["wav"])

    audio_path = None
    total_duration = 0.0

    if wav_bytes:
        audio_path, total_duration = save_wav_bytes_to_file(wav_bytes)
        st.success(f"Recording captured ({total_duration:.1f} s).")
        st.audio(wav_bytes, format="audio/wav")

    elif uploaded_wav is not None:
        # Enforce WAV to avoid ffmpeg dependency on Cloud
        bytes_data = uploaded_wav.read()
        audio_path, total_duration = save_wav_bytes_to_file(bytes_data)
        st.success(f"WAV file uploaded ({total_duration:.1f} s).")
        st.audio(bytes_data, format="audio/wav")

    # Analyze
    if st.button("🧠 Analyze") and audio_path:
        # Load ASR (cached)
        with st.status("Loading model (the first time can take a few seconds)…", expanded=False) as status:
            try:
                fw_model = load_fw_model()
                status.update(label="Model ready", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Model loading error: {e}")
                st.stop()

        # Transcribe
        with st.spinner("Transcribing…"):
            try:
                text = transcribe(fw_model, audio_path)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()

        st.markdown("### 📝 Transcription")
        st.write(text if text else "*(no transcription)*")

        if not text.strip():
            st.warning("No text to analyze.")
            st.stop()

        # Grammar/style analysis (LanguageTool)
        with st.spinner("Grammar & style analysis…"):
            try:
                lt_raw = call_languagetool(text, lt_endpoint)
                lt_matches = parse_lt_matches(lt_raw)
            except Exception as e:
                st.error(f"LanguageTool error: {e}")
                lt_matches = []

        st.markdown("### ✍️ Issues & suggestions")
        if not lt_matches:
            st.info("No significant issues found (or API limit reached).")
        else:
            for i, m in enumerate(lt_matches, 1):
                suggestion = m.replacements[0] if m.replacements else "—"
                with st.expander(f"#{i} {m.message}"):
                    st.markdown(f"- **Sentence:** {m.sentence}")
                    st.markdown(f"- **Suggestion:** `{suggestion}`")
                    st.markdown(f"- **Rule:** `{m.rule_id}`  • **Category:** `{m.category}`")

        corrected = apply_corrections(text, lt_matches)
        st.markdown("### ✅ Corrected version (auto-generated)")
        st.write(corrected)

        if auto_tts and corrected.strip():
            try:
                mp3 = tts_bytes_io(corrected, lang="it")
                st.audio(mp3, format="audio/mp3")
                st.download_button("⬇️ Download audio (MP3)", data=mp3, file_name="correction.mp3", mime="audio/mpeg")
            except Exception as e:
                st.warning(f"TTS unavailable: {e}")

        # Lightweight metrics (no librosa)
        text_metrics = compute_text_metrics(text)
        wpm = round(text_metrics.get("n_words", 0) / (max(total_duration, 1e-6) / 60.0), 1)

        st.markdown("### 📊 Indicators")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Words", text_metrics["n_words"])
            st.metric("Sentences", text_metrics["n_sentences"])
        with colB:
            st.metric("TTR (lexical diversity)", text_metrics["ttr"])
            st.metric("Avg. sentence length", text_metrics["avg_sentence_len"])
        with colC:
            st.metric("WPM (words/min)", wpm)
            st.metric("Level (estimate)", estimate_cefr(text_metrics))

st.markdown("---")
st.caption(
    "This build is optimized for Streamlit Cloud: it forces the tiny ASR model and accepts WAV input only. "
    "For heavier classroom usage, consider a self-hosted LanguageTool endpoint."
)
