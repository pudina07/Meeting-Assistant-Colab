"""
app.py  --  Streamlit TEST harness for the meeting pipeline.

    streamlit run app.py

Flow: upload -> [1] transcriber  (WhisperX: faster-whisper + word alignment, runs locally)
             -> [2] diarizer     (WhisperX diarization = pyannote community-1)   optional, needs HF_TOKEN
             -> [3] refiner      (Qwen2.5-7B-Instruct, LLM #1, domain-RAG glossary + edit validator)
             -> [4] extractor    (Qwen2.5-7B-Instruct, LLM #2, speaker-grounded owners)
             -> [4b] confidence  (deterministic evidence scorer + audio timestamps)
Shows raw / refined transcripts (diff + audit log), the retrieved domain context, speakers,
a decision / task review board with confidence tiers and click-to-listen timestamps,
minutes and JSON, and offers downloads.

Large recordings: the upload cap is raised in .streamlit/config.toml (run the app from this
folder). Files are streamed to disk, WhisperX windows the audio itself (nothing is uploaded anywhere),
and a small playback copy is kept for the review player. A local file path can be used instead of uploading.
"""
from __future__ import annotations

import difflib
import html
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from diarizer import DiarizationError, apply_diarization, diarization_available, diarize, speaker_stats
from confidence import TIER_HELP, TIER_ICON, TIERS
from extractor import MeetingRecord, extract_meeting_record, render_markdown
from refiner import LLMError, RefinementResult, refine_transcript
from transcriber import (MODEL_CHOICES, SUPPORTED_EXTENSIONS, TranscriptionError, TranscriptionResult, fmt_ts,
                         make_playback_audio, resolve_runtime, transcribe, whisperx_available)

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")


# ------------------------------------------------------------------ helpers
def diff_html(raw: str, refined: str) -> tuple[str, str]:
    """Word-level diff -> (raw_html, refined_html) with changed words highlighted."""
    a, b = raw.split(), refined.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ra, rb = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        ta, tb = html.escape(" ".join(a[i1:i2])), html.escape(" ".join(b[j1:j2]))
        if op == "equal":
            ra.append(ta)
            rb.append(tb)
        else:
            if ta:
                ra.append(f"<span style='background:#ffd6d6;text-decoration:line-through'>{ta}</span>")
            if tb:
                rb.append(f"<span style='background:#d4f5d4'>{tb}</span>")
    return " ".join(ra), " ".join(rb)


def build_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def _stage_input(source) -> tuple[str, str, bool]:
    """-> (path on disk, display name, is_temporary). `source` is a Streamlit UploadedFile or a local path."""
    if isinstance(source, (str, Path)):
        p = Path(str(source).strip().strip('"').strip("'")).expanduser()
        return str(p), p.stem, False
    suffix = Path(source.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        source.seek(0)
        shutil.copyfileobj(source, tmp, length=8 * 1024 * 1024)     # stream, don't duplicate in RAM
        return tmp.name, Path(source.name).stem, True


def run_pipeline(source, opts: dict):
    """Runs the stages in order. Returns a dict of results or raises a user-facing error."""
    tmp_path, name, is_temp = _stage_input(source)

    extra_warnings: list[str] = []
    diar_info = None
    try:
        with st.status("Processing recording...", expanded=True) as status:
            bar = st.progress(0.0)

            # ---- Stage 1: transcription
            rt = resolve_runtime(opts["wx_model"] if opts["wx_model"] != "auto" else None)
            st.write(f"**Stage 1 - Transcription** (WhisperX · {rt['model']} · {rt['device'].upper()} · {rt['compute_type']})")
            tr: TranscriptionResult = transcribe(
                tmp_path, vocab_hint=opts["domain_hints"],
                model=opts["wx_model"] if opts["wx_model"] != "auto" else None,
                progress_cb=lambda f, m: bar.progress(min(f, 1.0) * 0.35, text=m),
            )
            st.write(f"✅ {len(tr.segments)} segments, {tr.duration_s / 60:.1f} min of audio")
            playback = make_playback_audio(tmp_path)          # small copy for the click-to-listen player

            # ---- Stage 2: speaker diarization (optional, never fatal)
            if opts["diarize"]:
                st.write("**Stage 2 - Speaker diarization** (WhisperX · pyannote community-1)")
                try:
                    diar = diarize(
                        tmp_path, hf_token=opts["hf_token"] or None,
                        num_speakers=opts["num_speakers"] or None,
                        min_speakers=opts["min_speakers"] or None, max_speakers=opts["max_speakers"] or None,
                        progress_cb=lambda f, m: bar.progress(0.35 + 0.20 * min(f, 1.0), text=m),
                    )
                    apply_diarization(tr, diar)
                    diar_info = {"speakers": speaker_stats(tr), "device": diar.device, "model": diar.model}
                    st.write(f"✅ {len(diar_info['speakers'])} speaker(s) found (ran on {diar.device})")
                except DiarizationError as e:
                    extra_warnings.append(f"Speaker labels skipped: {e}")
                    st.write(f"⚠️ Skipped speaker labels: {e}")
            else:
                st.write("**Stage 2 - Speaker diarization:** off")

            # ---- Stage 3: refinement (LLM #1)
            st.write("**Stage 3 - Domain-aware refinement** (Qwen2.5-7B-Instruct, LLM #1)")
            rf: RefinementResult = refine_transcript(
                tr.turns(), domain_hints=opts["domain_hints"],
                use_dictionaries=opts["use_dictionaries"], use_keybert=opts["use_keybert"],
                use_briefing=opts["use_briefing"], api_key=opts["llm_key"] or None,
                base_url=opts["base_url"] or None, model=opts["llm_model"] or None,
                progress_cb=lambda f, m: bar.progress(0.55 + 0.25 * min(f, 1.0), text=m),
            )
            stt = rf.stats()
            st.write(f"✅ {stt['edits_applied']} edit(s) applied in {stt['turns_changed']} paragraph(s); "
                     f"{stt['edits_rejected']} proposed edit(s) rejected by the safety check")

            # ---- Stage 4: documentation (LLM #2)
            st.write("**Stage 4 - Meeting documentation** (Qwen2.5-7B-Instruct, LLM #2)")
            rec: MeetingRecord = extract_meeting_record(
                rf, duration_s=tr.duration_s, api_key=opts["llm_key"] or None,
                base_url=opts["base_url"] or None, model=opts["llm_model"] or None,
                progress_cb=lambda f, m: bar.progress(0.80 + 0.20 * min(f, 1.0), text=m),
            )
            st.write(f"✅ {len(rec.decisions)} decision/proposal item(s), {len(rec.action_items)} action item(s)")
            tiers = [x.confidence for x in [*rec.decisions, *rec.action_items] if x.confidence]
            if tiers:
                st.write("**Stage 4b - Evidence & confidence scoring:** "
                         + " · ".join(f"{TIER_ICON[t]} {tiers.count(t)} {t}" for t in TIERS if tiers.count(t)))
            bar.progress(1.0, text="Done")
            status.update(label="Processing complete", state="complete", expanded=False)
        return {"tr": tr, "rf": rf, "rec": rec, "diar": diar_info, "warnings": extra_warnings,
                "name": name, "audio": playback}
    finally:
        if is_temp:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def seek_to(seconds) -> None:
    """Button callback: move the review player to `seconds`."""
    if seconds is None:
        return
    target = int(max(0, seconds))
    if st.session_state.get("seek") == target:      # same target twice -> nudge so the player re-seeks
        target = max(0, target - 1)
    st.session_state["seek"] = target
    st.session_state["autoplay"] = True


def play_button(seconds, label: str, key: str, container=st) -> None:
    if seconds is None:
        container.caption("no timestamp")
        return
    container.button(f"▶ {label}", key=key, on_click=seek_to, args=(seconds,),
                     help="Play the recording from here (player in the sidebar)")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    st.caption("Leave blank to use values from your .env file.")
    llm_key = st.text_input("OpenRouter API key (Qwen 7B)", type="password", placeholder="sk-or-v1-...")
    with st.expander("Advanced LLM endpoint"):
        base_url = st.text_input("Base URL", placeholder="https://openrouter.ai/api/v1")
        llm_model = st.text_input("Model id", placeholder="qwen/qwen-2.5-7b-instruct")

    st.subheader("Transcription (WhisperX, local)")
    wx_ok, wx_why = whisperx_available()
    if not wx_ok:
        st.error(wx_why)
    else:
        auto_rt = resolve_runtime()
        st.caption(f"Detected: **{auto_rt['device'].upper()}** → automatic model: "
                   f"`{auto_rt['model']}` ({auto_rt['compute_type']}, batch {auto_rt['batch_size']})")
    wx_model = st.selectbox(
        "Whisper model", MODEL_CHOICES, index=0,
        help="auto = large-v3-turbo on a GPU, small.en on a CPU. Smaller = faster and lighter on a weak PC "
             "(tiny.en < base.en < small.en < medium.en < large-v3-turbo). The model downloads once on first use.")

    st.subheader("Speaker diarization")
    ok, why = diarization_available()
    do_diar = st.checkbox("Identify who is speaking (WhisperX diarization)",
                          value=bool(ok and os.getenv("HF_TOKEN")),
                          help="Labels each paragraph 'Speaker 1/2/...' so action-item owners can be grounded "
                               "('I will do it' -> the speaker who said it). This is the heaviest step on a CPU - "
                               "untick it for a quick run; everything else still works.")
    if not ok:
        st.caption(f"⚠️ {why}")
    hf_token = st.text_input("Hugging Face token", type="password", placeholder="hf_...",
                             help="Read token. You must also accept the terms on "
                                  "huggingface.co/pyannote/speaker-diarization-community-1 (same account).")
    c1, c2, c3 = st.columns(3)
    num_speakers = c1.number_input("Exact", 0, 20, 0, help="0 = detect automatically")
    min_speakers = c2.number_input("Min", 0, 20, 0)
    max_speakers = c3.number_input("Max", 0, 20, 0)

    st.subheader("Domain knowledge (RAG)")
    use_dictionaries = st.checkbox("Retrieve from built-in domain dictionaries", value=True)
    use_keybert = st.checkbox("Use KeyBERT for keyphrases (if installed)", value=True)
    use_briefing = st.checkbox("Let the LLM read the whole meeting first (briefing)", value=True)
    domain_hints = st.text_area(
        "Your own terms / names / notes (optional)",
        placeholder="Terms: Kubernetes, Postgres, OKRs\nsequel -> SQL\nPeople: Priya, Aarav, Rahul",
        height=110,
        help="Added to the glossary the refiner sees (and to Whisper's vocabulary prompt). "
             "Formats: 'Term', 'alias -> Term', or plain notes.",
    )

st.title("🎙️ AI Meeting Assistant")
st.caption("WhisperX (local transcription + word alignment + speakers) → domain-aware Qwen2.5-7B refiner → "
           "Qwen2.5-7B minutes & task extractor")

uploaded = st.file_uploader(
    "Upload a meeting recording",
    help="Supported: " + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))
         + ". Long recordings (e.g. a 25-min WAV) are fine - WhisperX windows them automatically.",
)
with st.expander("…or use a file already on this machine (very large recordings)"):
    local_path = st.text_input("Full path to an audio / video file", placeholder="C:/meetings/board_meeting.wav",
                               help="Skips the browser upload entirely - handy for multi-hundred-MB files.")
source = uploaded if uploaded is not None else (local_path.strip() or None)

if st.button("▶ Process recording", type="primary", disabled=source is None):
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    opts = dict(wx_model=wx_model, llm_key=llm_key, base_url=base_url, llm_model=llm_model,
                diarize=do_diar, hf_token=hf_token, num_speakers=int(num_speakers),
                min_speakers=int(min_speakers), max_speakers=int(max_speakers),
                use_dictionaries=use_dictionaries, use_keybert=use_keybert, use_briefing=use_briefing,
                domain_hints=domain_hints)
    try:
        st.session_state.pop("seek", None)
        st.session_state["result"] = run_pipeline(source, opts)
    except (TranscriptionError, LLMError) as e:      # LLMError covers RefinerError & ExtractionError
        st.session_state["error"] = str(e)
    except Exception as e:                            # never show a stack trace to the user
        st.session_state["error"] = f"Unexpected error: {type(e).__name__}: {e}"

if st.session_state.get("error"):
    st.error(f"❌ Processing failed - {st.session_state['error']}")

res = st.session_state.get("result")
if res:
    tr: TranscriptionResult = res["tr"]
    rf: RefinementResult = res["rf"]
    rec: MeetingRecord = res["rec"]
    base = res["name"]

    for w in [*tr.warnings, *res["warnings"], *rf.warnings]:
        st.warning(w)

    raw_txt = rf.raw_text()
    refined_txt = rf.text()
    raw_ts_txt = rf.raw_text(timestamps=True)
    refined_ts_txt = rf.text(timestamps=True)

    # ---------------- click-to-listen review player (sidebar, so it stays visible) -------------
    with st.sidebar:
        st.divider()
        st.subheader("🎧 Review player")
        if res.get("audio"):
            audio_bytes, mime = res["audio"]
            start = int(st.session_state.get("seek", 0))
            st.caption(f"Position: {fmt_ts(start)} - press ▶ next to any decision, task or paragraph to jump there.")
            try:
                st.audio(audio_bytes, format=mime, start_time=start,
                         autoplay=bool(st.session_state.pop("autoplay", False)))
            except TypeError:                                   # older Streamlit without autoplay
                st.audio(audio_bytes, format=mime, start_time=start)
        else:
            st.caption("Playback copy unavailable (ffmpeg without an MP3/Opus/AAC encoder). "
                       "Use the timestamps with your own player.")
    minutes_md = render_markdown(rec)
    record_json = rec.to_json()
    report_json = json.dumps(rf.report(), indent=2, ensure_ascii=False)

    tabs = st.tabs(["1 · Raw transcript", "2 · Refined transcript", "3 · Domain context",
                    "4 · Speakers", "5 · Decisions & tasks review", "6 · Meeting minutes", "7 · Structured JSON"])

    with tabs[0]:
        show_ts_raw = st.toggle("Show timestamps", value=True, key="ts_raw")
        st.text_area("Raw transcript (WhisperX output" + (", speaker-labelled" if tr.has_speakers else "") + ")",
                     raw_ts_txt if show_ts_raw else raw_txt, height=450, key="raw_view")

    with tabs[1]:
        mode = st.radio("View", ["Refined text", "Listen by paragraph", "Side-by-side diff",
                                 "Applied corrections", "Rejected edits"], horizontal=True)
        if mode == "Refined text":
            show_ts_ref = st.toggle("Show timestamps", value=True, key="ts_ref")
            st.text_area("Refined transcript", refined_ts_txt if show_ts_ref else refined_txt,
                         height=450, key="ref_view")
        elif mode == "Listen by paragraph":
            st.caption("Every paragraph with its start time and speech-recognition clarity. "
                       "Low clarity (< 0.5) means Whisper was unsure - worth a listen.")
            for i, t in enumerate(rf.turns):
                c1, c2 = st.columns([1, 9])
                play_button(t.start, fmt_ts(t.start), key=f"para_{i}", container=c1)
                clar = f" · clarity {t.asr_conf:.2f}" + (" ⚠" if t.asr_conf < 0.5 else "") \
                    if t.asr_conf is not None else ""
                tag = f"**{t.speaker}** " if t.speaker else ""
                c2.markdown(f"{tag}<small>{html.escape(t.text)}<br><span style='color:gray'>"
                            f"{fmt_ts(t.start)}–{fmt_ts(t.end)}{clar}</span></small>", unsafe_allow_html=True)
        elif mode == "Side-by-side diff":
            c1, c2 = st.columns(2)
            c1.markdown("**Raw**")
            c2.markdown("**Refined**")
            for t in rf.turns:
                a, b = diff_html(t.raw, t.text)
                tag = f"**{t.speaker}**  \n" if t.speaker else ""
                c1.markdown(f"{tag}<small>{a}</small>", unsafe_allow_html=True)
                c2.markdown(f"{tag}<small>{b}</small>", unsafe_allow_html=True)
        elif mode == "Applied corrections":
            kinds = sorted({e.kind for e in rf.applied if e.kind not in ("capitalisation", "punctuation")})
            show_minor = st.checkbox("Also show capitalisation / punctuation edits", value=False)
            shown = [e for e in rf.applied if show_minor or e.kind not in ("capitalisation", "punctuation")]
            if not shown:
                st.info("No substantive corrections were applied to this transcript.")
            for e in shown:
                st.markdown(f"**{e.kind}** · `{html.escape(e.before)}` → `{html.escape(e.after)}`  \n"
                            f"<small>{html.escape(e.reason)} · …{html.escape(e.context)}…</small>",
                            unsafe_allow_html=True)
        else:
            st.caption("Edits the LLM proposed but the safety check refused (numbers, negation, names, rewrites).")
            if not rf.rejected:
                st.info("Nothing was rejected.")
            for e in rf.rejected:
                st.markdown(f"`{html.escape(e.before)}` → `{html.escape(e.after)}`  \n"
                            f"<small>❌ {html.escape(e.reason)} · …{html.escape(e.context)}…</small>",
                            unsafe_allow_html=True)
        st.caption(json.dumps(rf.stats()))

    with tabs[2]:
        dc = rf.domain_context
        st.markdown("**Domains detected:** " + (", ".join(d["title"] for d in dc.get("active_domains", [])) or "none"))
        st.markdown(f"**Keyphrase extractor:** {dc.get('keyphrase_method', '-')}  ·  "
                    + ", ".join(dc.get("keyphrases", [])[:15]))
        if dc.get("briefing"):
            with st.expander("LLM briefing (whole-meeting read)"):
                st.json(dc["briefing"])
        st.markdown("**Glossary injected into the refiner's prompt**")
        st.dataframe(dc.get("glossary", []))
        st.markdown("**Suspect spans found in this transcript (heard → likely)**")
        st.dataframe(dc.get("suspect_spans", []))
        if dc.get("spelling_variants"):
            st.markdown("**Same word spelled several ways**")
            st.json(dc["spelling_variants"])
        if rf.unresolved:
            st.markdown("**Suspected, but the LLM left them unchanged**")
            st.dataframe(rf.unresolved)

    with tabs[3]:
        if res["diar"]:
            st.dataframe([{"speaker": k, **v} for k, v in res["diar"]["speakers"].items()])
            st.dataframe([{"speaker": s.label, "identified as": s.name, "evidence": s.evidence}
                          for s in rec.speakers])
        else:
            st.info("Speaker diarization was not run (or was skipped). Enable it in the sidebar and add an HF token.")

    with tabs[4]:
        st.caption("Each item is scored from the evidence in the recording. Press ▶ to hear it yourself.")
        st.markdown("  \n".join(f"{TIER_ICON[t]} **{t}** — {h}" for t, h in TIER_HELP.items()))
        pick = st.multiselect("Show confidence tiers", list(TIERS), default=list(TIERS), key="tier_filter")

        def _quote(q: str, ts: str) -> str:
            return f"> “{html.escape(q)}”" + (f"  `[{ts}]`" if ts else "") if q else ""

        st.markdown("### Decisions")
        shown_d = [d for d in rec.decisions if (d.confidence or "Low chance") in pick]
        if not shown_d:
            st.info("No decisions in the selected tiers.")
        for i, d in enumerate(shown_d):
            with st.container(border=True):
                c1, c2 = st.columns([7, 3])
                kind = "Agreed decision" if d.status == "agreed" else "Proposal (not agreed)"
                c1.markdown(f"**{html.escape(d.statement)}**  \n<small>{kind}</small>", unsafe_allow_html=True)
                c2.markdown(f"### {TIER_ICON.get(d.confidence, '')} {d.confidence}")
                c2.progress(d.confidence_score / 100, text=f"{d.confidence_score}% confidence")
                if d.source_quote:
                    c1.markdown(_quote(d.source_quote, d.timestamp))
                if d.agreement_quote:
                    c1.markdown("Settled by:\n" + _quote(d.agreement_quote, d.agreement_timestamp))
                b1, b2, _ = c1.columns([2, 2, 4])
                play_button(d.timestamp_s, f"Listen {d.timestamp}" if d.timestamp else "Listen",
                            key=f"dec_{i}", container=b1)
                if d.agreement_timestamp_s is not None and d.agreement_timestamp != d.timestamp:
                    play_button(d.agreement_timestamp_s, f"Settlement {d.agreement_timestamp}",
                                key=f"dec_agr_{i}", container=b2)
                with c2.expander("Why this score?"):
                    st.markdown("\n".join(f"- {html.escape(r)}" for r in d.confidence_reasons) or "-")

        st.markdown("### Action items")
        shown_a = [a for a in rec.action_items if (a.confidence or "Low chance") in pick]
        if not shown_a:
            st.info("No action items in the selected tiers.")
        for i, a in enumerate(shown_a):
            with st.container(border=True):
                c1, c2 = st.columns([7, 3])
                c1.markdown(f"**{html.escape(a.description)}**")
                c1.markdown(f"Owner: **{html.escape(a.owner)}**"
                            + (" _(first-person commitment)_" if a.owner_basis == "first_person" else "")
                            + f" · Deadline: **{html.escape(a.deadline)}** · Status: `{a.status}`")
                c2.markdown(f"### {TIER_ICON.get(a.confidence, '')} {a.confidence}")
                c2.progress(a.confidence_score / 100, text=f"{a.confidence_score}% confidence")
                if a.source_quote:
                    c1.markdown(_quote(a.source_quote, a.timestamp))
                play_button(a.timestamp_s, f"Listen {a.timestamp}" if a.timestamp else "Listen",
                            key=f"act_{i}", container=c1)
                with c2.expander("Why this score?"):
                    st.markdown("\n".join(f"- {html.escape(r)}" for r in a.confidence_reasons) or "-")

    with tabs[5]:
        st.markdown(minutes_md)

    with tabs[6]:
        st.json(json.loads(record_json))

    st.subheader("Downloads")
    d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
    d1.download_button("Raw transcript (.txt)", raw_txt, f"{base}_raw_transcript.txt", "text/plain")
    d2.download_button("Refined transcript (.txt)", refined_txt, f"{base}_refined_transcript.txt", "text/plain")
    d3.download_button("Minutes (.md)", minutes_md, f"{base}_minutes.md", "text/markdown")
    d4.download_button("Record (.json)", record_json, f"{base}_record.json", "application/json")
    d5.download_button("Refinement report (.json)", report_json, f"{base}_refinement_report.json", "application/json")
    d6.download_button("Timestamped transcript (.txt)", refined_ts_txt,
                       f"{base}_refined_transcript_timestamped.txt", "text/plain")
    d7.download_button(
        "All (.zip)",
        build_zip({
            f"{base}_raw_transcript.txt": raw_txt,
            f"{base}_refined_transcript.txt": refined_txt,
            f"{base}_raw_transcript_timestamped.txt": raw_ts_txt,
            f"{base}_refined_transcript_timestamped.txt": refined_ts_txt,
            f"{base}_minutes.md": minutes_md,
            f"{base}_record.json": record_json,
            f"{base}_refinement_report.json": report_json,
        }),
        f"{base}_outputs.zip", "application/zip",
    )
