# AI Meeting Assistant (v3: WhisperX + Qwen 7B)

Upload a meeting recording → transcript → domain-corrected transcript → minutes, decisions and action items,
with timestamps and a confidence tier for every decision / task.

```
audio ─▶ [1] WhisperX: Whisper transcription + word alignment      (local)
      ─▶ [2] WhisperX diarization: "Speaker 1 / 2 / …"             (local, optional, needs HF_TOKEN)
      ─▶ [3] LLM #1  Qwen2.5-7B-Instruct: domain-aware refinement   (OpenRouter)
      ─▶ [4] LLM #2  Qwen2.5-7B-Instruct: minutes, decisions, tasks (OpenRouter, separate prompt/contract)
      ─▶ [4b] deterministic grounding + confidence scoring (no LLM)
```

| Role | Model |
|---|---|
| Speech-to-text | WhisperX (faster-whisper; default `small.en` on CPU, `large-v3-turbo` on an NVIDIA GPU) + wav2vec2 word alignment |
| Speaker labels (optional) | WhisperX diarization step = pyannote `speaker-diarization-community-1` |
| LLM #1 – transcript refinement | `qwen/qwen-2.5-7b-instruct` |
| LLM #2 – minutes / decisions / tasks | `qwen/qwen-2.5-7b-instruct` (separate call, own prompt and output schema) |

Owners and deadlines are shown only when stated in the recording; otherwise they are `Unspecified`.
Every quote, owner and deadline is checked against the transcript in code after the LLM answers.

## Setup (once)

**Requirements:** Python 3.10 – 3.13, `ffmpeg` on PATH, an OpenRouter key, and (only for speaker labels) a Hugging Face token.

1. **ffmpeg** – Windows: `winget install ffmpeg` · macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg`.
   Open a new terminal afterwards and check `ffmpeg -version`.
2. **Fresh virtual environment** (do not reuse the old one: WhisperX needs pyannote-audio ≥ 4 and torch 2.8, which conflict with the previous pyannote 3.1 install):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux:  source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. **Install** (pick ONE):
   * *CPU only (no NVIDIA GPU), smallest download:*
     ```
     pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
     pip install -r requirements.txt
     ```
   * *NVIDIA GPU (CUDA 12.8):*
     ```
     pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
     pip install -r requirements.txt
     ```
   If installation fails on Python 3.13, use Python 3.11 or 3.12 for the venv.
4. **Keys** – copy `.env.example` to `.env` and fill in:
   * `OPENROUTER_API_KEY` – https://openrouter.ai/keys (the 7B model is very cheap).
   * `HF_TOKEN` – only for speaker labels: create a *read* token at https://huggingface.co/settings/tokens, then, logged in with the
     same account, open https://huggingface.co/pyannote/speaker-diarization-community-1 and click **Agree and access repository**.
5. *(Optional)* put your own terms in `domain_kb/custom/my_terms.txt`.

## Run

```
streamlit run app.py
```
Run it **from this folder** (the 2 GB upload limit lives in `.streamlit/config.toml`). Then: upload a file (or paste a local file path),
press **Process recording**, read the tabs, download the results.

The first run downloads the models (Whisper for the chosen size, the ~360 MB wav2vec2 aligner, and the speaker model if enabled);
later runs reuse the cache.

## Keeping it light on a modest PC

* Sidebar → **Whisper model**: `tiny.en` / `base.en` are fastest, `small.en` is the CPU default, `medium.en` and `large-v3-turbo` are
  more accurate but much slower on a CPU. A GPU is detected automatically.
* Untick **Identify who is speaking** for a fast run. Speaker identification is the heaviest step on a CPU; with it, "I'll do it"
  can be tied to a speaker, without it such tasks keep the owner `Unspecified`.
* Out of memory? In `.env`: `WHISPERX_BATCH=2` and/or a smaller `WHISPERX_MODEL`. To keep the PC responsive, lower `WHISPERX_THREADS`.
* Setting `Min` / `Max` speakers in the sidebar helps the diarizer when you know the head-count.

## Outputs
Raw transcript, refined transcript (plain and `[mm:ss]`-timestamped), minutes (`.md`), machine-readable record (`.json`, same decisions and
tasks as the minutes), refinement report (`.json`), and a `.zip` of everything.

## Troubleshooting
| Message | Fix |
|---|---|
| `ffmpeg/ffprobe not found` | install ffmpeg (step 1) and open a new terminal |
| `WhisperX could not be imported` | activate the venv; re-run step 3 |
| `Hugging Face refused access to the speaker model` | valid READ token **and** "Agree and access repository" on the community-1 page |
| `Not enough memory` | smaller model, `WHISPERX_BATCH=2`, or untick speakers |
| `LLM provider rejected the API key` / `402` | check `OPENROUTER_API_KEY` / add credits |
| A long `torchcodec` warning at start-up but the run succeeds | usually harmless – audio is decoded by ffmpeg |

`archive/verification.py` is the removed Chain-of-Verification stage (see `CHANGES.md` for how to re-enable it).
