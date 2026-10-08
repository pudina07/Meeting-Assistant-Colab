"""
transcriber.py  --  STAGE 1: speech-to-text  (WhisperX, runs LOCALLY)

Engine : WhisperX  (https://github.com/m-bain/whisperX)
           faster-whisper  -> batched Whisper transcription (CTranslate2, 4-70x faster than vanilla Whisper)
           VAD             -> cuts the recording into speech-only windows before Whisper sees it
           wav2vec2        -> forced alignment, i.e. accurate WORD-level timestamps
Input  : any meeting recording (mp3 / wav / m4a / flac / ogg / webm / mp4 ...)
Output : TranscriptionResult -> segments (+ word timings) and a clean, TIMESTAMP-FREE transcript

What this module does
---------------------
1. Validates the file (exists, supported extension, non-empty, decodable, has audio,
   not too short / too long) and raises a human-readable TranscriptionError otherwise.
2. Picks sensible defaults for THIS machine (GPU -> large-v3-turbo / float16, CPU -> small.en / int8),
   all overridable from .env or the app sidebar.
3. Runs WhisperX: transcribe -> forced alignment. No chunking / upload limits any more - audio never leaves
   the machine and WhisperX handles long recordings itself (VAD windows, batched).
4. Loads the models ONE AT A TIME and frees each one before the next stage (Whisper -> aligner -> diarizer),
   so peak RAM / VRAM is the largest single model, not the sum. This is what keeps a modest PC alive.
5. Keeps Whisper's per-segment avg_logprob -> an "ASR clarity" score per paragraph, used later by
   confidence.py to tell the user when a decision rests on a poorly-heard passage. (If the installed
   WhisperX does not return it, the mean word-alignment score is used as a stand-in.)
6. Drops obvious Whisper loops / hallucinations. (VAD already removes most silence hallucinations.)
7. build_turns(): groups segments into readable paragraphs / speaker turns - the unit that
   the refiner and extractor work on.

Speaker labels are added by diarizer.py (WhisperX's diarization step) - this module is transcription only.

Requires: ffmpeg + ffprobe on PATH, `pip install whisperx python-dotenv`   (see requirements.txt)
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration  (every value can be overridden in .env; "auto" = pick for this machine)
# --------------------------------------------------------------------------------------
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "auto")        # auto | tiny.en | base.en | small.en | medium.en | large-v3-turbo | large-v3 ...
WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "auto")      # auto | cuda | cpu
WHISPERX_COMPUTE = os.getenv("WHISPERX_COMPUTE", "auto")    # auto | float16 | int8_float16 | int8 | float32
WHISPERX_BATCH = os.getenv("WHISPERX_BATCH", "auto")        # auto | integer
WHISPERX_VAD = os.getenv("WHISPERX_VAD", "pyannote")        # pyannote | silero
WHISPERX_THREADS = os.getenv("WHISPERX_THREADS", "auto")    # CPU threads; auto = all cores but one (max 8)

MODEL_CHOICES = ["auto", "tiny.en", "base.en", "small.en", "medium.en", "distil-large-v3", "large-v3-turbo", "large-v3"]

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".webm",
    ".mp4", ".mpeg", ".mpga", ".aac", ".wma", ".mkv", ".mov",
}
MIN_DURATION_S = 1.0
MAX_DURATION_S = 2 * 60 * 60          # 2 hours cap
SAMPLE_RATE = 16000

ProgressCB = Optional[Callable[[float, str], None]]


class TranscriptionError(Exception):
    """Raised with a message that is safe to show directly to the end user."""


# --------------------------------------------------------------------------------------
# Data classes  (unchanged: refiner / extractor / confidence / app all rely on these)
# --------------------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None          # "Speaker 1", ... (filled in by diarizer.apply_diarization)
    words: list[Word] = field(default_factory=list)
    avg_logprob: Optional[float] = None    # Whisper's mean token log-probability for this segment


@dataclass
class Turn:
    """A readable paragraph: one speaker's continuous talk (or a pause-delimited paragraph when
    diarization is off). Timings are kept only for internal alignment and never displayed."""
    speaker: Optional[str]
    text: str
    start: float = 0.0
    end: float = 0.0
    asr_conf: Optional[float] = None       # 0..1 Whisper clarity (exp of duration-weighted avg_logprob)


def logprob_to_conf(lp: Optional[float]) -> Optional[float]:
    if lp is None:
        return None
    return round(max(0.0, min(1.0, math.exp(lp))), 3)


def fmt_ts(seconds: Optional[float]) -> str:
    """12.3 -> '00:12', 3725 -> '1:02:05'."""
    if seconds is None:
        return "--:--"
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


_SENT_END = re.compile(r"[.!?][\"')\]]*$")


def build_turns(segments: list[Segment], max_chars: int = 900, soft_chars: int = 600,
                pause_s: float = 2.5) -> list[Turn]:
    """Group Whisper fragments into paragraphs.
    * a new turn starts when the speaker changes (if known) or after a long pause;
    * long turns are split at a sentence end once they pass `soft_chars` (hard cap `max_chars`)."""
    turns: list[Turn] = []
    cur: Optional[Turn] = None
    acc: list[float] = [0.0, 0.0]          # [sum(lp * dur), sum(dur)] for the current turn

    def _close():
        if cur is not None:
            cur.asr_conf = logprob_to_conf(acc[0] / acc[1]) if acc[1] > 0 else None
            turns.append(cur)

    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        new = (
            cur is None
            or seg.speaker != cur.speaker
            or seg.start - cur.end > pause_s
            or len(cur.text) + 1 + len(t) > max_chars
            or (len(cur.text) >= soft_chars and _SENT_END.search(cur.text) is not None)
        )
        if new:
            _close()
            cur = Turn(seg.speaker, t, seg.start, seg.end)
            acc = [0.0, 0.0]
        else:
            cur.text += " " + t
            cur.end = seg.end
        if seg.avg_logprob is not None:
            dur = max(0.2, seg.end - seg.start)
            acc[0] += seg.avg_logprob * dur
            acc[1] += dur
    _close()
    return turns


def render_turns(turns: list[Turn], timestamps: bool = False) -> str:
    """Plain readable text: 'Speaker 1: ...' blocks when speakers are known, otherwise paragraphs.
    timestamps=True prefixes each block with its start time, e.g. '[03:12] Speaker 2: ...'."""
    blocks = []
    for t in turns:
        b = f"{t.speaker}: {t.text}" if t.speaker else t.text
        if timestamps:
            b = f"[{fmt_ts(getattr(t, 'start', None))}] {b}"
        blocks.append(b)
    return "\n\n".join(blocks)


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    duration_s: float
    model: str
    language: str = "en"
    chunks: int = 1                         # kept for compatibility; WhisperX does its own windowing
    has_word_timings: bool = False
    warnings: list[str] = field(default_factory=list)
    device: str = "cpu"

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    def turns(self) -> list[Turn]:
        return build_turns(self.segments)

    def text(self, timestamps: bool = False) -> str:
        """Transcript (speaker-labelled when diarization has been applied); timestamps optional."""
        return render_turns(self.turns(), timestamps=timestamps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------------------
def _require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise TranscriptionError(
            "ffmpeg/ffprobe not found on this machine. Install it first "
            "(Windows: `winget install ffmpeg`; macOS: `brew install ffmpeg`; "
            "Ubuntu/Debian: `sudo apt install ffmpeg`) and restart the app."
        )


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TranscriptionError("Audio conversion timed out. The file may be too large or damaged.")


def validate_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise TranscriptionError("The uploaded file could not be found.")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise TranscriptionError(
            f"Unsupported file type '{p.suffix or 'unknown'}'. Supported formats: "
            + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))
            + "."
        )
    if p.stat().st_size == 0:
        raise TranscriptionError("The file is empty (0 bytes). Please upload a valid recording.")
    return p


def probe_audio(path: Path) -> Optional[float]:
    """Return duration in seconds (None if the container doesn't report it).
    Raises TranscriptionError when the file is unreadable or has no audio stream."""
    proc = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
        timeout=120,
    )
    if proc.returncode != 0:
        raise TranscriptionError(
            "The file could not be read as audio. It may be corrupted or not a real media file."
        )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise TranscriptionError("The file could not be analysed as audio (corrupted?).")
    if not info.get("streams"):
        raise TranscriptionError("The file contains no audio track.")
    try:
        return float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None


def make_playback_audio(src: str | Path, max_seconds: Optional[float] = None) -> Optional[tuple[bytes, str]]:
    """Small mono copy of the recording for the in-app 'jump to timestamp' player
    (a 25 min meeting -> ~6 MB instead of a 260 MB WAV). Returns (bytes, mime) or None."""
    if not (shutil.which("ffmpeg")):
        return None
    src = Path(src)
    with tempfile.TemporaryDirectory(prefix="play_") as tmp:
        for ext, codec, mime in (("mp3", ["-c:a", "libmp3lame", "-b:a", "48k"], "audio/mpeg"),
                                 ("ogg", ["-c:a", "libopus", "-b:a", "32k"], "audio/ogg"),
                                 ("m4a", ["-c:a", "aac", "-b:a", "48k"], "audio/mp4")):
            dst = Path(tmp) / f"playback.{ext}"
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "22050"]
            if max_seconds:
                cmd += ["-t", f"{max_seconds:.0f}"]
            try:
                proc = subprocess.run(cmd + codec + [str(dst)], capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                return None
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return dst.read_bytes(), mime
    return None


# --------------------------------------------------------------------------------------
# WhisperX runtime helpers
# --------------------------------------------------------------------------------------
def whisperx_available() -> tuple[bool, str]:
    """(ok, reason). Cheap check that does not load any model."""
    try:
        import torch  # noqa: F401
        import whisperx  # noqa: F401
    except Exception as e:   # ImportError, or a broken torch / ctranslate2 install
        return False, (f"WhisperX is not installed or failed to import ({type(e).__name__}: {e}). "
                       "Run: pip install -r requirements.txt")
    return True, ""


def _import_whisperx():
    try:
        import whisperx
        return whisperx
    except Exception as e:
        raise TranscriptionError(
            f"WhisperX could not be imported ({type(e).__name__}: {e}). Install the dependencies in a "
            "fresh virtual environment with `pip install -r requirements.txt` (Python 3.10 - 3.13)."
        )


def call_with_supported_kwargs(fn: Callable, *args, **kwargs):
    """Call fn, silently dropping keyword arguments the installed WhisperX version does not know.
    (WhisperX renamed / added a few parameters between releases, e.g. use_auth_token -> token.)"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def detect_device(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested in ("cpu", "cuda"):
        if requested == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available - falling back to CPU")
                    return "cpu"
            except Exception:
                return "cpu"
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _vram_gb() -> float:
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return 0.0


def resolve_runtime(model: Optional[str] = None, device: Optional[str] = None,
                    compute: Optional[str] = None, batch: Optional[str | int] = None) -> dict[str, Any]:
    """Turn 'auto' settings into concrete ones for this machine.
        GPU : large-v3-turbo, float16 (int8_float16 if < 6 GB VRAM), batch 16 / 8 / 4 by VRAM
        CPU : small.en, int8, batch 4  (large-v3-turbo also runs on CPU - just much slower)"""
    dev = detect_device(device or WHISPERX_DEVICE)
    model = (model or WHISPERX_MODEL or "auto").strip()
    compute = (compute or WHISPERX_COMPUTE or "auto").strip().lower()
    batch = str(batch if batch not in (None, "") else WHISPERX_BATCH).strip().lower()

    vram = _vram_gb() if dev == "cuda" else 0.0
    if model == "auto":
        model = "large-v3-turbo" if dev == "cuda" else "small.en"
    if compute == "auto":
        compute = ("int8_float16" if vram and vram < 6 else "float16") if dev == "cuda" else "int8"
    if dev == "cpu" and compute in ("float16", "int8_float16"):      # CPUs cannot run fp16
        compute = "int8"
    if batch == "auto":
        bs = (16 if vram >= 10 else 8 if vram >= 6 else 4) if dev == "cuda" else 4
    else:
        try:
            bs = max(1, int(batch))
        except ValueError:
            bs = 4
    threads = WHISPERX_THREADS.lower()
    if threads == "auto":
        n = max(1, min(8, (os.cpu_count() or 4) - 1))
    else:
        try:
            n = max(1, int(threads))
        except ValueError:
            n = 4
    return {"model": model, "device": dev, "compute_type": compute, "batch_size": bs, "threads": n}


def release_memory() -> None:
    """Free Python + GPU memory between WhisperX stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


def _prob_to_logprob(p: float) -> float:
    return math.log(max(1e-3, min(1.0, p)))


def _convert_segments(aligned_segments: list[dict]) -> tuple[list[Segment], bool, bool]:
    """WhisperX segments (dicts) -> our Segment objects.
    Returns (segments, any_word_timings, used_alignment_score_as_clarity)."""
    out: list[Segment] = []
    any_words = False
    used_proxy = False
    for s in aligned_segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        try:
            s_start = float(s.get("start", 0.0) or 0.0)
            s_end = float(s.get("end", s_start) or s_start)
        except (TypeError, ValueError):
            continue

        words: list[Word] = []
        scores: list[float] = []
        raw_words = s.get("words") or []
        prev_end = s_start
        for w in raw_words:
            wt = (w.get("word") or "").strip()
            if not wt:
                continue
            ws, we = w.get("start"), w.get("end")
            ws = float(ws) if ws is not None and not (isinstance(ws, float) and math.isnan(ws)) else prev_end
            we = float(we) if we is not None and not (isinstance(we, float) and math.isnan(we)) else max(ws, prev_end)
            if we < ws:
                we = ws
            prev_end = we
            words.append(Word(round(ws, 2), round(we, 2), wt))
            sc = w.get("score")
            if sc is not None and not (isinstance(sc, float) and math.isnan(sc)):
                scores.append(float(sc))
        if words:
            any_words = True

        lp = s.get("avg_logprob")
        if isinstance(lp, (list, tuple)):
            lp = lp[0] if lp else None
        if lp is None and scores:                   # older WhisperX: no avg_logprob -> use alignment confidence
            lp = _prob_to_logprob(sum(scores) / len(scores))
            used_proxy = True
        out.append(Segment(round(s_start, 2), round(max(s_end, s_start), 2), text, words=words,
                           avg_logprob=float(lp) if lp is not None else None))
    return out, any_words, used_proxy


def _clean(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Drop runaway repeats (Whisper loops) and low-confidence one-liners (typical silence hallucinations)."""
    cleaned: list[Segment] = []
    prev_norm: Optional[str] = None
    repeat = dropped = 0
    for seg in sorted(segments, key=lambda x: x.start):
        n = _norm(seg.text)
        repeat = repeat + 1 if n == prev_norm else 0
        prev_norm = n
        if repeat >= 2:                                          # same line 3+ times in a row = loop
            dropped += 1
            continue
        if seg.avg_logprob is not None and seg.avg_logprob < -1.5 and len(seg.text.split()) <= 3:
            dropped += 1
            continue
        cleaned.append(seg)
    return cleaned, dropped


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def transcribe(
    audio_path: str | Path,
    vocab_hint: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    batch_size: Optional[int] = None,
    progress_cb: ProgressCB = None,
) -> TranscriptionResult:
    """
    Transcribe a meeting recording with WhisperX (transcription + forced word alignment).

    vocab_hint: optional comma-separated domain terms / names ("Kubernetes, Rahul, Q3 OKRs").
                Passed to Whisper as its `initial_prompt`, which biases the spelling of those words.
    model / device / compute_type / batch_size: leave None to use .env / automatic choices.
    """
    report = progress_cb or (lambda f, m: None)

    path = validate_file(audio_path)
    _require_ffmpeg()
    report(0.01, "Checking audio file...")
    probed = probe_audio(path)
    if probed is not None:
        if probed < MIN_DURATION_S:
            raise TranscriptionError("The audio is too short (under 1 second) to contain a meeting.")
        if probed > MAX_DURATION_S:
            raise TranscriptionError(
                f"The recording is {probed/3600:.1f} h long; the maximum supported length is 2 hours.")

    whisperx = _import_whisperx()
    rt = resolve_runtime(model, device, compute_type, batch_size)
    dev, name, compute, bs = rt["device"], rt["model"], rt["compute_type"], rt["batch_size"]
    warnings: list[str] = []
    if dev == "cpu":
        try:
            import torch
            torch.set_num_threads(rt["threads"])
        except Exception:
            pass

    # ---- decode (ffmpeg, any container) -> float32 mono 16 kHz numpy array
    report(0.03, "Decoding audio...")
    try:
        audio = whisperx.load_audio(str(path))
    except Exception as e:
        raise TranscriptionError(f"The audio could not be decoded ({type(e).__name__}). The file may be damaged.")
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_DURATION_S:
        raise TranscriptionError("The audio is too short (under 1 second) to contain a meeting.")
    if duration > MAX_DURATION_S:
        raise TranscriptionError(
            f"The recording is {duration/3600:.1f} h long; the maximum supported length is 2 hours.")

    prompt = None
    if vocab_hint and vocab_hint.strip():
        prompt = ("Meeting vocabulary: " + vocab_hint.strip())[:800]
    asr_options = {"initial_prompt": prompt} if prompt else None

    # ---- 1) Whisper (faster-whisper, batched) ------------------------------------------------
    report(0.05, f"Loading Whisper '{name}' ({dev}, {compute})... first run downloads the model")
    asr = None
    try:
        asr = call_with_supported_kwargs(
            whisperx.load_model, name, dev, compute_type=compute, language="en",
            asr_options=asr_options, vad_method=WHISPERX_VAD, threads=rt["threads"])
        report(0.10, "Transcribing (speech windows are processed in batches)...")
        result = call_with_supported_kwargs(
            asr.transcribe, audio, batch_size=bs, language="en",
            progress_callback=lambda pct: report(0.10 + 0.65 * min(pct, 100.0) / 100.0,
                                                 f"Transcribing... {pct:.0f}%"))
    except TranscriptionError:
        raise
    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            raise TranscriptionError(
                f"Not enough {'GPU' if dev == 'cuda' else ''} memory for Whisper '{name}'. In the sidebar pick a "
                "smaller model (e.g. small.en) or set WHISPERX_BATCH=2 in .env, then run again.")
        raise TranscriptionError(f"Whisper transcription failed: {type(e).__name__}: {msg[:300]}")
    finally:
        del asr
        release_memory()

    segments = result.get("segments") or []
    if not segments:
        raise TranscriptionError(
            "No speech was detected in this recording. Check that the file contains audible English speech.")

    # ---- 2) forced alignment -> word timestamps (needed for accurate speaker boundaries) ------
    report(0.78, "Aligning words to the audio (wav2vec2)...")
    aligned_segments = segments
    align_model = None
    try:
        align_model, meta = whisperx.load_align_model(language_code="en", device=dev)
        aligned = call_with_supported_kwargs(
            whisperx.align, segments, align_model, meta, audio, dev, return_char_alignments=False,
            progress_callback=lambda pct: report(0.78 + 0.19 * min(pct, 100.0) / 100.0,
                                                 f"Aligning words... {pct:.0f}%"))
        aligned_segments = aligned.get("segments") or segments
    except Exception as e:   # not fatal: we still have segment-level timings
        logger.warning("WhisperX alignment failed: %s", e)
        warnings.append(f"Word-level alignment was skipped ({type(e).__name__}); timestamps are per segment "
                        "only, so speaker changes in the middle of a sentence may be missed.")
    finally:
        del align_model
        release_memory()

    converted, has_words, used_proxy = _convert_segments(aligned_segments)
    cleaned, dropped = _clean(converted)
    if dropped:
        warnings.append(f"Removed {dropped} low-confidence / repeated segment(s) likely caused by silence or noise.")
    if used_proxy:
        warnings.append("This WhisperX version does not report Whisper's own confidence, so 'clarity' scores are "
                        "estimated from the word-alignment scores.")
    if not cleaned:
        raise TranscriptionError(
            "No speech was detected in this recording. Check that the file contains audible English speech.")

    report(1.0, "Transcription complete.")
    return TranscriptionResult(
        segments=cleaned, duration_s=float(duration), model=f"WhisperX / {name} ({compute})",
        chunks=1, has_word_timings=has_words, warnings=warnings, device=dev,
    )


if __name__ == "__main__":      # quick manual test:  python transcriber.py meeting.mp3
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>\n(For the full app run:  streamlit run app.py)")
        raise SystemExit(1)
    try:
        res = transcribe(sys.argv[1], progress_cb=lambda f, m: print(f"{f:5.0%}  {m}"))
    except TranscriptionError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    print(res.text())
