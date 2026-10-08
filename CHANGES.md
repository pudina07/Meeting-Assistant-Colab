# What changed in v3 (WhisperX · Qwen 7B · no verification stage)

## 1. WhisperX replaces Groq Whisper + the standalone pyannote script (Stages 1 & 2)
| | before | now |
|---|---|---|
| Speech-to-text | Whisper large-v3-turbo on Groq (cloud, 25 MB upload limit, chunk + re-encode logic) | **WhisperX** on this machine: faster-whisper (batched) + VAD + wav2vec2 word alignment |
| Speakers | `pyannote/speaker-diarization-3.1` called directly | **WhisperX diarization step** (`whisperx.diarize.DiarizationPipeline`, pyannote `speaker-diarization-community-1` inside) |
| API keys | `GROQ_API_KEY` + `HF_TOKEN` | only `HF_TOKEN` (and only if you want speaker labels) |

* `transcriber.py` is rewritten around WhisperX. All chunking, ffmpeg re-encoding, retry-on-rate-limit and Groq code is gone (nothing is uploaded any more). File validation, error messages, `Segment/Turn/build_turns`, the playback copy and the timestamp helpers are unchanged, so refiner / extractor / confidence need no changes.
* `diarizer.py` now calls WhisperX's diarization step. The word-level speaker assignment, flicker smoothing, segment splitting and "Speaker N" numbering (`apply_diarization`) are the same code as before.
* **Lighter on RAM**: Whisper, the aligner and the diarizer are loaded one at a time and freed before the next one (peak = largest single model, not the sum).
* **Machine-aware defaults** (override in `.env` or the sidebar): NVIDIA GPU -> `large-v3-turbo` / float16; CPU -> `small.en` / int8; batch size and CPU threads picked automatically (one core is left free).
* ASR "clarity" for the confidence scores still comes from Whisper's `avg_logprob` (WhisperX returns it). If an older WhisperX does not, the mean word-alignment score is used instead and a note is shown.
* Silence hallucinations are mostly removed by WhisperX's VAD; the repeat-loop and low-confidence one-liner filters remain.
* Speaker identification is now **off by default unless `HF_TOKEN` is set** (it is the heaviest step on a CPU).
* The code tolerates different WhisperX releases (unknown keyword arguments are dropped, `token` vs `use_auth_token`), and a failed alignment download degrades to segment-level timings instead of aborting.

## 2. Qwen2.5-72B-Instruct -> Qwen2.5-7B-Instruct (Stages 3 & 4)
* Default model id is now `qwen/qwen-2.5-7b-instruct` on OpenRouter (`LLM_MODEL` in `.env`; the 72B id still works if you ever want it back).
* The 7B model has a 32k-token window and is weaker on very long prompts, so the one-pass limit for minutes extraction is now 30 000 characters (was 48 000) and long meetings are split into 20 000-character parts (was 36 000). Both are env-tunable (`EXTRACT_SINGLE_PASS_CHARS`, `EXTRACT_CHUNK_CHARS`).
* Everything that protects against a smaller model's mistakes is unchanged and still runs: the refiner's edit validator (numbers / negation / names are protected), verbatim-quote checks, owner/deadline grounding, and the deterministic confidence scorer.

## 3. Hallucination Chain-of-Verification removed (old Stage 4c)
* `extractor.py` no longer calls it, and the duplicate grounding + scoring pass that followed it is gone too (the same checks already run once, right after extraction).
* `MeetingRecord` no longer carries `verification_enabled` / `verification_findings`; the "7 · Verification" tab and the verification download were removed from `app.py`.
* The module is kept as `archive/verification.py`. To bring it back: move it next to `extractor.py`, re-add the import, the two fields on `MeetingRecord` and the `verify_and_revise(...)` call (followed by `ground_record` + `score_record`) in `extract_meeting_record`, and the tab/download in `app.py`.

---

# Earlier versions

## 1. Confidence tiers for decisions and action items (new Stage 4b, `confidence.py`)
LLM #2 still decides *what* the decisions/tasks are, but it no longer has the last word on *how sure* we are.
It now also reports **evidence features** it can read off the text:

| item | features |
|---|---|
| decision | `agreement_signal` (explicit / implicit / none), `objection` (none / resolved / unresolved), `hedged` |
| action item | `acceptance` (explicit / implicit / none), `hedged` |

A deterministic scorer (no LLM) combines them with hard evidence from the pipeline into a 0–100 score:
- the evidence quote is really in the transcript (an invented quote is capped at 30, i.e. Low chance)
- lexical cues in the quote and the next few turns: formal settlement ("motion carried", "that's settled"), assent, push-back ("I'm not convinced", "hold on"), hedges ("maybe", "probably"), first-person commitments ("I'll …"). These **cross-check the LLM**: an "explicit" claim with no agreement wording nearby is downgraded to implicit.
- acknowledgement by a *different* speaker (when diarization is on)
- **ASR clarity** of the passage (Whisper `avg_logprob`, now kept per paragraph)
- whether the refiner changed words inside the quote

| tier | score | meaning |
|---|---|---|
| 🟢 Confirmed | ≥ 80 | explicitly settled / accepted, verified quote, clear audio |
| 🔵 High chance | 60–79 | agreed or accepted, but implicitly or with one weaker signal |
| 🟠 Ambiguous | 40–59 | mixed signals: push-back, hedging, unclear audio, or a well-supported proposal |
| 🔴 Low chance | < 40 | only suggested or weakly evidenced |

The scorer also enforces hard rules so a score can never contradict the problem statement: proposals are capped at Ambiguous, "Confirmed" needs explicit settlement (or explicit acceptance by a stated owner), an agreed decision with an open objection is forced to Ambiguous, unclear audio caps an item at High chance, and scores never go above 97%. Every point added or removed is stored as a readable reason ("Why this score?").

## 2. Timestamps on every decision and action item
- Paragraph timings and clarity are carried from the transcriber through the refiner (`RefinedTurn.start/end/asr_conf`).
- Each quote is located in the transcript and its start time is interpolated inside the paragraph. Decisions get a statement timestamp and a separate **settlement timestamp**.
- Streamlit: a **review player** in the sidebar (always visible). Every ▶ button (decisions, tasks, and each paragraph in "Listen by paragraph") jumps the player to 2 s before that point. It plays a small 48 kbps copy of the recording (about 9 MB for 25 min).
- New tab **"5 · Decisions & tasks review"** with tier filter, confidence bar, quotes with times, and listen buttons.
- Transcripts can be shown or downloaded with `[mm:ss]` prefixes. Minutes markdown and JSON include `timestamp`, `confidence`, `confidence_score`, `confidence_reasons` and `clarity`.

## 3. Large-file fix (25-min recordings)  — _the chunking / re-encoding part is obsolete since v3: WhisperX runs locally and has no upload limit_
- Root cause: Streamlit's default **200 MB upload cap**. A 25-min WAV is about 260 MB, so the upload was refused before the pipeline ran. `.streamlit/config.toml` raises it to 2 GB. **Run `streamlit run app.py` from this folder** so the config is picked up.
- Uploads are streamed to disk instead of copied in memory.
- Adaptive chunking: each 10-min chunk is size-checked. If the FLAC is over the STT limit it is re-encoded to 64 then 32 kbps MP3 (or Opus), and as a last resort split in half, recursively. Overlap ownership guarantees no duplicate or missing text.
- New "use a file already on this machine" path input, so very large files skip the browser upload entirely.
- Tested on a 25-min, 264 MB stereo WAV: 3 chunks of 7–14 MB with the default limit, and 10 parts when forced to a 1 MB limit, with identical segments in both runs.


## 4. Transcript-grounded Chain-of-Verification (Stage 4c)  — _REMOVED in v3, code kept in archive/verification.py_
- Draft summary/minutes/decisions/tasks are converted into atomic claims.
- Each claim is independently verified against retrieved transcript evidence; the verifier never receives the draft record.
- A final conservative revision removes or weakens unsupported claims, while existing deterministic grounding is run again afterward.
- Verification results, evidence quotes and timestamps are shown in a dedicated UI tab and exported as JSON.
- Verification calls are factored and parallelized (up to 6 workers) to keep wall-clock latency reasonable.
