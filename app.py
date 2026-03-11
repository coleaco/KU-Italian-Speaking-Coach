# ---------------------------------------------------------
# Italian Speaking Practice — Custom Rules Edition
# - ASR model selector (tiny/base/small)
# - Multi-format uploader (WAV/MP3/M4A/OGG/WEBM)
# - Word-level timestamps + low-confidence highlighting
# - Italian apostrophe normalization
# - Free feedback: LanguageTool (+ public endpoint) + Custom A2–B1 rules
#   * Plural noun endings (agreement with plural determiners)
#   * Prepositions with cities, countries/regions (articulated), common places
#   * Year with seasons (di -> del), "spend time" (speso/spento -> trascorso)
# - Optional TTS for corrected text
# ---------------------------------------------------------

import os
import io
import time
import json
import wave
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

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

# For custom rules
@dataclass
class Issue:
    message: str
    start: int
    end: int
    suggestion: str
    rule_id: str
    category: str = "Custom"

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
    # Clean stray spaces and duplicated spaces
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

def parse_lt_matches(data: Dict) -> List[LTMatch]:
    out: List[LTMatch] = []
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
# ---- Custom Rules (A2–B1) --
# ---------------------------

TOKEN_RE = re.compile(r"\b[\wàèéìòóùÀÈÉÌÒÓÙ]+\b", re.UNICODE)

# 1) Year with seasons: "... estate di 2024" -> "... estate del 2024"
SEASON_YEAR_PAT = re.compile(
    r"\b(estat[ea]|invern[oa]|primaver[ae]|autunn[oa])\s+di\s+(20\d{2})\b",
    re.IGNORECASE
)

# 2) Spend time: "ho speso/spento X settimane" -> "ho trascorso X settimane"
SPEND_TIME_PAT = re.compile(
    r"\b(ho|hai|ha|abbiamo|avete|hanno)\s+(?:spes\w+|spento)\s+((?:\w+\s+){0,2}?(?:giorno|giorni|settimana|settimane|mese|mesi|anno|anni))\b",
    re.IGNORECASE
)

def year_preposition_issues(text: str) -> List[Issue]:
    issues: List[Issue] = []
    for m in SEASON_YEAR_PAT.finditer(text):
        start, end = m.start(), m.end()
        repl = f"{m.group(1)} del {m.group(2)}"
        issues.append(Issue(
            message="Usa 'del' davanti all'anno in questo contesto (non 'di').",
            start=start, end=end, suggestion=repl,
            rule_id="IT_YEAR_PREP"
        ))
    return issues

def spend_time_issues(text: str) -> List[Issue]:
    issues: List[Issue] = []
    for m in SPEND_TIME_PAT.finditer(text):
        start, end = m.start(), m.end()
        aux, duration = m.group(1), m.group(2)
        repl = f"{aux} trascorso {duration}"
        issues.append(Issue(
            message="Per il tempo si usa 'trascorrere' (o 'passare'), non 'spendere'.",
            start=start, end=end, suggestion=repl,
            rule_id="IT_SPEND_TIME"
        ))
    return issues

# 3) Countries/Regions canonical forms (articulated prepositions)
COUNTRY_FORMS = {
    "Italia": "in Italia",
    "Francia": "in Francia",
    "Spagna": "in Spagna",
    "Germania": "in Germania",
    "Svizzera": "in Svizzera",
    "Austria": "in Austria",
    "Grecia": "in Grecia",
    "Portogallo": "in Portogallo",
    "Regno Unito": "nel Regno Unito",
    "Inghilterra": "in Inghilterra",
    "Scozia": "in Scozia",
    "Irlanda": "in Irlanda",
    "Stati Uniti": "negli Stati Uniti",
    "USA": "negli Stati Uniti",
    "Paesi Bassi": "nei Paesi Bassi",
    "Olanda": "nei Paesi Bassi",
    "Cina": "in Cina",
    "Giappone": "in Giappone",
    "Corea": "in Corea",
    "India": "in India",
    "Australia": "in Australia",
    "Canada": "in Canada",
    "Messico": "in Messico",
    "Brasile": "in Brasile",
    "Marocco": "in Marocco",
    "Egitto": "in Egitto",
    "Turchia": "in Turchia",
    "Sicilia": "in Sicilia",
    "Sardegna": "in Sardegna",
    "Toscana": "in Toscana",
    "Lombardia": "in Lombardia",
}

# 4) Common places/buildings canonical forms
PLACE_FORMS = {
    "scuola": "a scuola",
    "casa": "a casa",
    "lavoro": "al lavoro",
    "ristorante": "al ristorante",
    "bar": "al bar",
    "supermercato": "al supermercato",
    "parco": "al parco",
    "università": "all'università",
    "stadio": "allo stadio",
    "stazione": "alla stazione",
    "fermata": "alla fermata",
    "festa": "alla festa",
    "aeroporto": "all'aeroporto",
    "biblioteca": "in biblioteca",
    "palestra": "in palestra",
    "farmacia": "in farmacia",
    "banca": "in banca",
    "spiaggia": "in spiaggia",
    "montagna": "in montagna",
    "campagna": "in campagna",
    "centro": "in centro",
    "ufficio": "in ufficio",
    "chiesa": "in chiesa",
}

# 5) City rule (simple): "in Roma/Milano/Firenze" -> "a Roma/..."
#    We will skip this if the location matches one of the COUNTRY_FORMS keys (avoid conflicts).
CITY_IN_PAT = re.compile(r"\bin\s+([A-ZÀ-Ý][a-zà-ÿ]+)\b")

PREP_VARIANTS = r"(?:a|in|al|allo|alla|all'|all’|nel|nello|nella|nell'|nell’|nei|negli|nelle)"

def keep_title_case(suggestion: str, original_token: str) -> str:
    if original_token and original_token[0].isupper():
        # Capitalize only the token part (last word)
        parts = suggestion.split()
        if parts:
            parts[-1] = parts[-1].capitalize()
            return " ".join(parts)
    return suggestion

def location_preposition_issues(text: str) -> List[Issue]:
    issues: List[Issue] = []

    lower_text = text.lower()

    # 5a) Countries/regions: wrong preposition before a known name
    for name, canonical in COUNTRY_FORMS.items():
        # Build pattern: any preposition variant + optional articles before the name
        # Handle multi-word names (e.g., "Stati Uniti", "Regno Unito")
        name_pat = re.escape(name)
        pat = re.compile(rf"\b{PREP_VARIANTS}\s+{name_pat}\b", re.IGNORECASE)
        for m in pat.finditer(text):
            span = text[m.start():m.end()]
            # If it's already canonical, skip
            if span.lower() == canonical.lower():
                continue
            # Suggest canonical form, preserving capitalization of the location head word
            suggestion = canonical
            # Preserve capitalization of the last word of the name
            last_word = name.split()[-1]
            suggestion = keep_title_case(suggestion, last_word)
            issues.append(Issue(
                message=f"Con paesi/regioni usa ‘{canonical.split()[0]}’ (forma articolata se richiesta).",
                start=m.start(), end=m.end(),
                suggestion=suggestion,
                rule_id="IT_LOC_COUNTRY"
            ))

        # Also catch bare "a Italia" or "alla Italia" not covered by PREP_VARIANTS variants
        # (mostly redundant, but harmless)

    # 5b) Common places/buildings: enforce canonical mapping
    for noun, canonical in PLACE_FORMS.items():
        # Match any preposition + (optional article) + noun (with or without article/apostrophe)
        noun_pat = r"(?:l'|lo|la|il|i|gli|le)?\s*" + re.escape(noun)
        pat = re.compile(rf"\b{PREP_VARIANTS}\s+{noun_pat}\b", re.IGNORECASE)
        for m in pat.finditer(text):
            span = text[m.start():m.end()]
            if span.lower() == canonical.lower():
                continue
            # Preserve capitalization if noun is capitalized in text
            # (rare for common places, but just in case)
            original_tail = text[m.start():m.end()].split()[-1]
            suggestion = canonical
            suggestion = keep_title_case(suggestion, original_tail)
            issues.append(Issue(
                message="Preposizione fissa con il luogo (usa la forma più naturale).",
                start=m.start(), end=m.end(),
                suggestion=suggestion,
                rule_id="IT_LOC_PLACE"
            ))

    # 5c) Cities: "in X" -> "a X" if X not in country/region list
    country_heads = {k.split()[-1] for k in COUNTRY_FORMS.keys()}
    for m in CITY_IN_PAT.finditer(text):
        city = m.group(1)
        # Skip if this looks like a country/region head (e.g., "Italia")
        if city in country_heads or city in COUNTRY_FORMS:
            continue
        suggestion = f"a {city}"
        issues.append(Issue(
            message="Con le città usa ‘a’, non ‘in’.",
            start=m.start(), end=m.end(),
            suggestion=suggestion,
            rule_id="IT_LOC_CITY"
        ))

    return issues

# 6) Plural noun endings with plural determiners
PLURAL_DETS = {
    "i", "gli", "le",
    "dei", "degli", "delle",
    "quei", "quegli", "quelle",
    "questi", "queste",
    "alcuni", "alcune",
    "molti", "molte",
    "tanti", "tante",
}

INVARIABLE = {
    "città", "università", "foto", "auto", "cinema", "computer", "bar"
}

LEX_EXCEPTIONS = {
    # masculine in -a class and typical exceptions:
    "amico": "amici",
    "greco": "greci",
    "medico": "medici",
    "psicologo": "psicologi",
    "biologo": "biologi",
    "teologo": "teologi",
    "catalogo": "cataloghi",
    "dialogo": "dialoghi",
    "lago": "laghi",
    "mano": "mani",
    "problema": "problemi",
    "programma": "programmi",
    "schema": "schemi",
    "tema": "temi",
    "poeta": "poeti",
}

def is_all_lower_or_title(word: str) -> bool:
    # Allow lowercase or Title Case; skip ALL-CAPS (likely acronyms)
    return not word.isupper()

def regular_plural_candidate(sg: str) -> str:
    if sg in LEX_EXCEPTIONS:
        return LEX_EXCEPTIONS[sg]
    if sg in INVARIABLE:
        return sg
    if sg.endswith("o"):
        return sg[:-1] + "i"
    if sg.endswith("a"):
        return sg[:-1] + "e"
    if sg.endswith("e"):
        return sg[:-1] + "i"
    return sg

def plural_noun_issues(text: str) -> List[Issue]:
    issues: List[Issue] = []
    tokens = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    for idx, (tok, t_start, t_end) in enumerate(tokens[:-1]):
        det = tok.lower()
        if det not in PLURAL_DETS:
            continue
        next_tok, n_start, n_end = tokens[idx + 1]
        noun = next_tok
        if len(noun) <= 2:
            continue
        if not is_all_lower_or_title(noun):
            continue
        lower = noun.lower()
        if lower in INVARIABLE:
            continue
        if lower.endswith(("o", "a", "e")):
            plural_guess = regular_plural_candidate(lower)
            if plural_guess != lower:
                if noun[0].isupper():
                    plural_suggestion = plural_guess.capitalize()
                else:
                    plural_suggestion = plural_guess
                msg = ("Con i determinanti plurali usa il sostantivo al plurale "
                       f"(qui: “{noun}” → “{plural_suggestion}”).")
                issues.append(Issue(
                    message=msg,
                    start=n_start, end=n_end,
                    suggestion=plural_suggestion,
                    rule_id="IT_PLURAL_ENDING"
                ))
    return issues

def custom_check(text: str) -> List[Issue]:
    issues: List[Issue] = []
    issues.extend(year_preposition_issues(text))
    issues.extend(spend_time_issues(text))
    issues.extend(location_preposition_issues(text))
    issues.extend(plural_noun_issues(text))

    # De-duplicate overlapping issues by (span, rule, suggestion)
    dedup = []
    seen = set()
    for iss in issues:
        key = (iss.start, iss.end, iss.rule_id, iss.suggestion)
        if key not in seen:
            seen.add(key)
            dedup.append(iss)
    return dedup

def apply_issues(text: str, issues: List[Issue]) -> str:
    out = text
    for iss in sorted(issues, key=lambda x: x.start, reverse=True):
        if 0 <= iss.start <= len(out) and 0 <= iss.end <= len(out):
            out = out[:iss.start] + iss.suggestion + out[iss.end:]
    return out

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
    page_title="Italian Speaking Practice — Custom Rules",
    page_icon="🇮🇹",
    layout="wide"
)

st.title("🇮🇹 Italian Speaking Practice — Custom Rules Edition")

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
    input_duration = 0.0  # header-based for WAV; otherwise derive from segments

    # Recorder path (always WAV)
    if wav_bytes:
        audio_path = save_bytes_to_file(wav_bytes, ".wav")
        input_duration = wav_duration_seconds(audio_path)
        st.success(f"Recording captured ({input_duration:.1f}s).")
        st.audio(wav_bytes, format="audio/wav")

    # Uploaded file path (keep original suffix)
    elif uploaded:
        file_bytes = uploaded.read()
        suffix = os.path.splitext(uploaded.name)[1].lower() or ".wav"
        if suffix.lstrip(".") not in accepted_types:
            st.error("Unsupported file type. Please upload WAV, MP3, M4A, OGG, or WEBM.")
            st.stop()
        audio_path = save_bytes_to_file(file_bytes, suffix)
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

        # --- Grammar analysis (LanguageTool + Custom) ---
        word_count = len([w for w in text.split() if w.strip()])
        should_call_lt = word_count >= 5

        now = time.time()
        if now - st.session_state["last_lt_call_ts"] < LT_MIN_INTERVAL_SEC:
            should_call_lt = False
        else:
            st.session_state["last_lt_call_ts"] = now

        lt_result = {"ok": False, "rate_limited": False, "data": {}}
        matches: List[LTMatch] = []

        # Custom rules (always available, run locally)
        custom_issues = custom_check(text)

        with st.spinner("Analyzing grammar…"):
            if should_call_lt:
                lt_result = call_languagetool(text, lt_endpoint)
                if lt_result["ok"]:
                    matches = parse_lt_matches(lt_result["data"])
                else:
                    matches = []

        st.markdown("### ✍️ Issues & suggestions")

        # Custom issues first
        if custom_issues:
            for i, iss in enumerate(custom_issues, 1):
                with st.expander(f"[Custom] #{i} {iss.message}"):
                    bad_span = text[iss.start:iss.end]
                    st.write(f"**Span:** `{screenshot_safe(bad_span)}`")
                    st.write(f"**Suggestion:** `{screenshot_safe(iss.suggestion)}`")
                    st.write(f"Rule: `{iss.rule_id}` • Category: `Custom`")

        # LanguageTool issues next
        if lt_result.get("rate_limited"):
            st.warning("LanguageTool public API limit reached. Try again soon.")
        elif not should_call_lt and not custom_issues:
            if word_count < 5:
                st.info("Speak a bit more to get grammar/style feedback (≥ 5 words).")
            else:
                st.info("Analyzing… please try again in a moment.")
        elif not matches and not custom_issues:
            st.info("No significant grammar/style issues found.")
        else:
            for i, m in enumerate(matches, 1):
                suggestion = m.replacements[0] if m.replacements else "—"
                with st.expander(f"[LT] #{i} {m.message}"):
                    st.write(f"**Sentence:** {m.sentence}")
                    st.write(f"**Suggestion:** `{screenshot_safe(suggestion)}`")
                    st.write(f"Rule: `{m.rule_id}` • Category: `{m.category}`")

        # Apply both corrections to produce the corrected version
        corrected = normalize_italian_text(apply_corrections(text, matches))
        corrected = normalize_italian_text(apply_issues(corrected, custom_issues))

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
    "Custom rules cover: plurals with plural determiners; prepositions for cities, countries/regions, and common places; "
    "season+year (‘del 2024’); and ‘trascorrere’ for time spent. You can extend exceptions and whitelists as needed."
)
