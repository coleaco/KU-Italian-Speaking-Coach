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
