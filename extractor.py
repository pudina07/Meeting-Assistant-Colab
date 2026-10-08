"""
extractor.py  --  STAGE 3: meeting documentation  (LLM #2)

Model : Qwen2.5-7B-Instruct (a SEPARATE call with its own system prompt / output contract)
Input : refined transcript (RefinementResult | list[Turn] | plain text). When speaker diarization
        was run, every paragraph is labelled "Speaker N: ...", which lets the model ground owners
        of first-person commitments ("I'll send it by Friday") in the speaker who said them.
Output: MeetingRecord (pydantic)  --  ONE canonical object, rendered two ways:
          record.to_json()            machine-readable
          render_markdown(record)     human-readable
        Both therefore contain exactly the same decisions and tasks.

Anti-hallucination, in two layers
---------------------------------
1. Prompt: explicit definitions of agreed-vs-proposed and assigned-vs-unassigned, "Unspecified"
   defaults, verbatim evidence quote required for every decision / task.
2. Code (`ground_record`): deterministic checks AFTER the LLM answers
     - evidence quotes must really occur in the transcript    -> else item is flagged / downgraded
     - an owner must be a name that appears in the transcript, OR a "Speaker N" label whose own
       turn contains the first-person commitment               -> else "Unspecified"
     - a deadline's words must appear in the transcript        -> else "Unspecified"
     - "agreed" / "assigned" status is downgraded when the evidence does not support it
     - speaker names are accepted only with a verifiable quote from the transcript
3. Calibration (`confidence.score_record`, Stage 4b): the LLM only reports evidence FEATURES
   (agreement_signal / objection / hedged / acceptance); a deterministic scorer turns them, plus
   quote location, lexical cues, speaker turn-taking and ASR clarity, into a 0-100 score and a tier
   (Confirmed / High chance / Ambiguous / Low chance) with the reasons, and attaches the audio
   timestamp of every decision and action item so the user can listen and confirm.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from confidence import TIER_HELP, TIER_ICON, score_record
from refiner import LLMError, chat_json, get_model, make_llm_client
from transcriber import Turn

logger = logging.getLogger(__name__)

UNSPECIFIED = "Unspecified"
ProgressCB = Optional[Callable[[float, str], None]]
# A 7B model follows the long JSON contract best on moderate inputs (its window is 32k tokens in total,
# and the reply alone can use 4k). ~30k chars is about 7-8k tokens: a 25-30 min meeting fits in ONE pass,
# longer meetings are split into ~20k-char parts and merged. Raise these if you switch to a bigger model.
MAX_SINGLE_PASS_CHARS = int(os.getenv("EXTRACT_SINGLE_PASS_CHARS", "30000"))
CHUNK_CHARS = int(os.getenv("EXTRACT_CHUNK_CHARS", "20000"))


class ExtractionError(LLMError):
    pass


# --------------------------------------------------------------------------------------
# Canonical schema
# --------------------------------------------------------------------------------------
class Decision(BaseModel):
    statement: str
    status: Literal["agreed", "proposed"] = "proposed"
    source_quote: str = ""            # the words that state the decision / motion
    agreement_quote: str = ""         # the words that show it was settled ("carried", "agreed", "no objections")
    participants: list[str] = Field(default_factory=list)
    quote_verified: bool = False
    # evidence features reported by the LLM (inputs to confidence.py, never trusted on their own)
    agreement_signal: Literal["explicit", "implicit", "none"] = "none"
    objection: Literal["none", "resolved", "unresolved"] = "none"
    hedged: bool = False
    # filled in by confidence.score_record (never by the LLM)
    timestamp: str = ""                       # "12:34" where the decision is stated
    timestamp_s: Optional[float] = None       # seek position for the audio player
    agreement_timestamp: str = ""             # where it was settled ("agreed", "carried", ...)
    agreement_timestamp_s: Optional[float] = None
    confidence: str = ""                      # Confirmed | High chance | Ambiguous | Low chance
    confidence_score: int = 0                 # 0-100
    confidence_reasons: list[str] = Field(default_factory=list)
    clarity: Optional[float] = None           # Whisper clarity of the evidence passage (0..1)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("agreed", "proposed") else "proposed"      # conservative default

    @field_validator("agreement_signal", mode="before")
    @classmethod
    def _sig(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("explicit", "implicit", "none") else "none"

    @field_validator("hedged", mode="before")
    @classmethod
    def _hedged(cls, v):
        return str(v).strip().lower() in ("true", "1", "yes") if not isinstance(v, bool) else v

    @field_validator("objection", mode="before")
    @classmethod
    def _obj(cls, v):
        if isinstance(v, bool):
            return "unresolved" if v else "none"
        v = str(v or "").strip().lower()
        return v if v in ("none", "resolved", "unresolved") else "none"


class ActionItem(BaseModel):
    description: str
    owner: str = UNSPECIFIED
    owner_basis: Literal["named", "first_person", "none"] = "none"   # how the owner was established
    deadline: str = UNSPECIFIED
    status: Literal["assigned", "proposed", "unspecified"] = "unspecified"
    source_quote: str = ""
    quote_verified: bool = False
    acceptance: Literal["explicit", "implicit", "none"] = "none"     # LLM evidence feature
    hedged: bool = False                                                # LLM evidence feature
    # filled in by confidence.score_record (never by the LLM)
    timestamp: str = ""
    timestamp_s: Optional[float] = None
    confidence: str = ""
    confidence_score: int = 0
    confidence_reasons: list[str] = Field(default_factory=list)
    clarity: Optional[float] = None

    @field_validator("hedged", mode="before")
    @classmethod
    def _hedged(cls, v):
        return str(v).strip().lower() in ("true", "1", "yes") if not isinstance(v, bool) else v

    @field_validator("acceptance", mode="before")
    @classmethod
    def _acc(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("explicit", "implicit", "none") else "none"

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _unspec(cls, v):
        v = str(v).strip() if v is not None else ""
        return v or UNSPECIFIED

    @field_validator("owner_basis", mode="before")
    @classmethod
    def _basis(cls, v):
        v = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
        return v if v in ("named", "first_person", "none") else "none"

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("assigned", "proposed", "unspecified") else "unspecified"


class SpeakerInfo(BaseModel):
    label: str                              # "Speaker 2"
    name: str = UNSPECIFIED                 # real name / title, only if the transcript reveals it
    evidence: str = ""                      # verbatim quote that reveals it
    verified: bool = False

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v):
        v = str(v).strip() if v is not None else ""
        return v or UNSPECIFIED


class MinutesSection(BaseModel):
    heading: str
    points: list[str] = Field(default_factory=list)


class MeetingRecord(BaseModel):
    summary: str = ""
    key_topics: list[str] = Field(default_factory=list)
    minutes: list[MinutesSection] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    participants_detected: list[str] = Field(default_factory=list)
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    speaker_labels_used: bool = False
    meeting_duration_minutes: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    model: str = ""

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a meticulous meeting secretary. You turn a meeting transcript into structured minutes.
You must reflect ONLY what was actually said. Inventing anything is a serious failure.

__SPEAKER_RULES__

DEFINITIONS
  DECISION, status "agreed": the group explicitly settled something ("we've decided", "let's go with X",
     "agreed", "motion carried", "no objections", "that is unanimous", clear consensus with no objection).
  DECISION, status "proposed": a suggestion, option, preference or idea that was raised but NOT clearly settled
     ("we should probably", "what if we", "I think we could", "maybe", unresolved debate, a motion that was
     moved but whose vote is not in the transcript). Never label a proposal "agreed". When unsure, use "proposed".
  ACTION ITEM: concrete work somebody must do after the meeting.
     status "assigned"    : a named person (or the speaker themself, for a first-person commitment) explicitly takes it.
     status "proposed"    : someone suggests it but nobody commits ("we should look into X").
     status "unspecified" : it is clearly needed/agreed as work, but nobody is named.
  owner    : see the owner rules above; otherwise "Unspecified".
  deadline : ONLY if a time limit is stated for THAT task. Copy it as spoken ("by Friday", "next sprint",
     "end of the month", "in two weeks"). Do NOT convert to calendar dates. Otherwise "Unspecified".
  source_quote: for EVERY decision and action item, copy 1-2 consecutive sentences (max ~30 words) EXACTLY
     from the transcript that state it. No labels, no paraphrasing, no ellipses.
  agreement_quote (decisions only): copy the words that show the matter was settled (for example
     "That is unanimous and so carried."). Empty string when the decision was only proposed.

EVIDENCE FEATURES (report what the text shows; do not try to be optimistic):
  agreement_signal (decisions): "explicit" = someone clearly states it is decided / agreed / carried, or every
     person who responds clearly says yes; "implicit" = mild assent ("okay", "sure") or nobody objected and the
     meeting moved on, but nobody explicitly settled it; "none" = no sign of agreement.
  objection (decisions): "unresolved" = someone pushed back / raised a concern that was not settled;
     "resolved" = push-back was raised and then settled; "none" = nobody objected.
  hedged (decisions and action items): true when the wording is tentative ("maybe", "probably", "I think we
     could", "for now", "let's see", "we'll try").
  acceptance (action items): "explicit" = the owner clearly accepted ("I'll do it", "Sure, I'll handle that",
     or a direct assignment answered with "yes / will do"); "implicit" = assigned and not refused, but no clear
     acknowledgement; "none" = nobody accepted it.

OUTPUT: ONE JSON object, no markdown, no commentary, with exactly these keys:
{
  "summary": "2-4 sentence factual summary",
  "key_topics": ["short topic", ...],
  "minutes": [{"heading": "topic or phase of the meeting", "points": ["concise factual bullet", ...]}],
  "decisions": [{"statement": "...", "status": "agreed" | "proposed", "source_quote": "...", "agreement_quote": "...", "participants": ["names only if stated"], "agreement_signal": "explicit" | "implicit" | "none", "objection": "none" | "resolved" | "unresolved", "hedged": true | false}],
  "action_items": [{"description": "verb-first task", "owner": "Name / Speaker N / Unspecified", "owner_basis": "named" | "first_person" | "none", "deadline": "as spoken or Unspecified", "status": "assigned" | "proposed" | "unspecified", "source_quote": "...", "acceptance": "explicit" | "implicit" | "none", "hedged": true | false}],
  "participants_detected": ["names of people who are mentioned or addressed in the transcript"],
  "speakers": [{"label": "Speaker 1", "name": "real name/title or Unspecified", "evidence": "verbatim quote that reveals it"}]
}
Rules for the content:
  - "minutes": 3-8 sections in chronological order; every bullet must be traceable to the transcript.
  - Use empty lists when there are no decisions / tasks. Do not pad. "speakers" is [] when there are no speaker labels.
  - Do not merge separate tasks and do not split one task into several.
  - Keep numbers, names and negations exactly as in the transcript."""

SPEAKER_RULES_LABELLED = """\
The transcript is labelled with speakers ("Speaker 1:", "Speaker 2:", ...), produced automatically by a
diarization model. Use the labels to resolve WHO is speaking:
  - owner_basis "named"       : a person is NAMED as responsible ("Rahul will send the report", "Priya, can you
                                handle the deck?" followed by agreement). owner = that name exactly as spoken.
  - owner_basis "first_person": the speaker commits themself ("I'll send it", "I can take that", "let me handle it")
                                -> owner = the speaker's real name if the transcript reveals it (see "speakers"),
                                otherwise the label, e.g. "Speaker 2". The quote MUST contain the first-person words
                                and MUST be inside that speaker's own paragraph.
  - Never guess an owner from role, topic, or who spoke last. A request that nobody accepted has no owner.
  - "speakers": for each label, give a real name/title ONLY when the transcript reveals it: the person introduces
    themself ("this is Councillor Miller"), or the next speaker addresses them by name right after they spoke
    ("Thank you, Councillor Miller"). Copy the revealing sentence verbatim into "evidence". Otherwise "Unspecified".
    Labels can occasionally be wrong at speaker changes; if an attribution looks inconsistent, use "Unspecified"."""

SPEAKER_RULES_PLAIN = """\
The transcript has NO speaker labels. Therefore:
  - "I'll do it" / "I can take that" / "me" cannot be attributed to a person -> owner is "Unspecified", owner_basis "none".
  - An owner is allowed ONLY when a person is NAMED as the one responsible, e.g. "Rahul will send the report",
    "Priya, can you handle the deck?" followed by agreement. owner_basis is then "named".
  - Never guess an owner from role, topic, or who spoke last."""

MERGE_PROMPT = """\
You combine partial summaries of consecutive parts of ONE meeting.
Reply with ONE JSON object: {"summary": "2-4 sentence factual summary of the whole meeting",
"key_topics": ["deduplicated short topics"]}.
Use only information present in the partial summaries. Do not invent anything."""


# --------------------------------------------------------------------------------------
# Grounding / verification helpers
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
_PLACEHOLDERS = {"", "none", "n/a", "na", "null", "not specified", "unspecified", "unknown", "tbd",
                 "not stated", "nobody", "no one", "-"}
_NOT_NAMES = {"i", "me", "we", "us", "you", "he", "she", "they", "someone", "somebody", "everyone",
              "everybody", "anyone", "speaker", "team", "unknown", "unspecified", "user", "participant"}
_DEADLINE_STOP = {"by", "on", "before", "the", "of", "at", "in", "next", "this", "end", "a", "an", "to",
                  "until", "till", "within", "from", "then", "and", "or", "for", "after", "around",
                  "about", "latest", "that", "coming", "early", "late", "mid"}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("\u2019", "'").replace("'", ""))


def _ngrams(ws: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


_SPEAKER_LABEL = re.compile(r"^\s*speaker\s*(\d+)\s*$", re.I)
_LINE_LABEL = re.compile(r"^(Speaker \d+):\s*(.*)$")
_FIRST_PERSON = {"i", "ill", "im", "ive", "id", "me", "my", "myself", "let", "ll"}


class _Grounder:
    def __init__(self, turns: list[tuple[Optional[str], str]]):
        self.turns = turns
        plain = " ".join(t for _, t in turns)
        self.words = _words(plain)
        self.vocab = set(self.words)
        self.speakers = {sp for sp, _ in turns if sp}
        self._grams: dict[int, set] = {}
        self._turn_words = [_words(t) for _, t in turns]

    def _g(self, n: int) -> set:
        if n not in self._grams:
            self._grams[n] = _ngrams(self.words, n)
        return self._grams[n]

    def quote_ok(self, quote: str) -> bool:
        qw = _words(quote)
        if len(qw) < 3:
            return False
        n = min(4, len(qw))
        grams = _ngrams(qw, n)
        if not grams:
            return False
        hit = sum(1 for g in grams if g in self._g(n))
        return hit / len(grams) >= 0.7

    def quote_speakers(self, quote: str) -> set[str]:
        """Speaker labels whose OWN paragraph contains (most of) the quote."""
        qw = _words(quote)
        if len(qw) < 3:
            return set()
        n = min(4, len(qw))
        grams = _ngrams(qw, n)
        found: set[str] = set()
        for (sp, _), tw in zip(self.turns, self._turn_words):
            if sp and grams and sum(1 for g in grams if g in _ngrams(tw, n)) / len(grams) >= 0.7:
                found.add(sp)
        return found

    def owner_ok(self, owner: str) -> bool:
        if owner.strip().lower() in _PLACEHOLDERS:
            return False
        toks = [t for t in _words(owner) if t not in ("and", "the", "of")]
        if not toks or any(t in _NOT_NAMES for t in toks):
            return False
        return all(t in self.vocab for t in toks)

    def deadline_ok(self, deadline: str) -> bool:
        if deadline.strip().lower() in _PLACEHOLDERS:
            return False
        toks = [t for t in _words(deadline) if t not in _DEADLINE_STOP]
        return bool(toks) and all(t in self.vocab for t in toks)

    def name_ok(self, name: str) -> bool:
        toks = _words(name)
        return bool(toks) and all(t in self.vocab for t in toks)


def _label_of(owner: str) -> Optional[str]:
    m = _SPEAKER_LABEL.match(owner or "")
    return f"Speaker {int(m.group(1))}" if m else None


def ground_record(rec: MeetingRecord, transcript: Any) -> MeetingRecord:
    """Deterministic post-checks. Mutates and returns `rec`.
    `transcript` may be a string, a list of Turn, or a RefinementResult."""
    turns, _, _ = _prepare(transcript)
    g = _Grounder(turns)
    w = rec.warnings
    rec.speaker_labels_used = bool(g.speakers)

    # ---- speakers: names only with verifiable evidence -----------------------------------
    verified_names: dict[str, str] = {}
    by_label: dict[str, SpeakerInfo] = {}
    for sp in rec.speakers:
        lab = _label_of(sp.label)
        if not lab or lab not in g.speakers or lab in by_label:
            continue
        sp.label = lab
        if sp.name != UNSPECIFIED:
            toks = _words(sp.name)
            ev = set(_words(sp.evidence))
            if g.name_ok(sp.name) and g.quote_ok(sp.evidence) and toks and toks[-1] in ev and lab in g.quote_speakers(sp.evidence) | _neighbours(g, sp.evidence, lab):
                sp.verified = True
                verified_names[lab] = sp.name
            else:
                w.append(f"Name '{sp.name}' for {lab} could not be verified in the transcript -> kept as Unspecified.")
                sp.name, sp.evidence = UNSPECIFIED, ""
        by_label[lab] = sp
    for lab in sorted(g.speakers, key=lambda x: int(x.split()[1])):
        by_label.setdefault(lab, SpeakerInfo(label=lab))
    rec.speakers = list(by_label.values())

    # ---- decisions ----------------------------------------------------------------------
    for d in rec.decisions:
        src_ok, agr_ok = g.quote_ok(d.source_quote), g.quote_ok(d.agreement_quote)
        d.quote_verified = src_ok or agr_ok
        if d.status == "agreed" and not d.quote_verified:
            d.status = "proposed"
            w.append(f"Decision downgraded to 'proposed' (evidence quote not found in transcript): {d.statement[:80]}")
        elif d.status == "agreed" and d.agreement_quote and not agr_ok:
            d.agreement_quote = ""
        if d.status == "proposed":
            d.agreement_quote = d.agreement_quote if agr_ok else ""
        d.participants = [p for p in d.participants if g.name_ok(p)]

    # ---- action items -------------------------------------------------------------------
    for a in rec.action_items:
        a.quote_verified = g.quote_ok(a.source_quote)
        if not a.quote_verified:
            if a.owner != UNSPECIFIED or a.deadline != UNSPECIFIED or a.status == "assigned":
                w.append(f"Owner/deadline cleared (evidence quote not found in transcript): {a.description[:80]}")
            a.owner, a.deadline, a.owner_basis = UNSPECIFIED, UNSPECIFIED, "none"
            if a.status == "assigned":
                a.status = "unspecified"

        if a.owner != UNSPECIFIED:
            lab = _label_of(a.owner)
            qwords = set(_words(a.source_quote))
            speaks = g.quote_speakers(a.source_quote)
            first_person = bool(qwords & _FIRST_PERSON)
            if lab:
                if lab in speaks and first_person:
                    a.owner_basis = "first_person"
                    a.owner = f"{verified_names[lab]} ({lab})" if lab in verified_names else lab
                else:
                    w.append(f"Owner '{a.owner}' not supported by a first-person commitment inside that speaker's own turn "
                             f"-> Unspecified: {a.description[:60]}")
                    a.owner, a.owner_basis = UNSPECIFIED, "none"
            elif not g.owner_ok(a.owner):
                w.append(f"Owner '{a.owner}' is not a name found in the transcript -> set to Unspecified: {a.description[:60]}")
                a.owner, a.owner_basis = UNSPECIFIED, "none"
            elif a.owner_basis == "first_person":
                owner_tokens = set(_words(a.owner))
                spk = [l for l, n in verified_names.items() if owner_tokens & set(_words(n))]
                if not (first_person and spk and spk[0] in speaks):
                    if owner_tokens <= qwords:
                        a.owner_basis = "named"
                    else:
                        w.append(f"Owner '{a.owner}' could not be tied to the speaker of the commitment -> Unspecified: {a.description[:60]}")
                        a.owner, a.owner_basis = UNSPECIFIED, "none"
            else:
                a.owner_basis = "named"
        else:
            a.owner_basis = "none"

        if a.deadline != UNSPECIFIED and not g.deadline_ok(a.deadline):
            w.append(f"Deadline '{a.deadline}' not found in the transcript -> set to Unspecified: {a.description[:60]}")
            a.deadline = UNSPECIFIED
        if a.status == "assigned" and a.owner == UNSPECIFIED:
            a.status = "unspecified"

    # de-duplicate (identical wording after normalisation)
    def _dedupe(items, key):
        seen, out = set(), []
        for it in items:
            k = " ".join(_words(key(it)))
            if k and k not in seen:
                seen.add(k)
                out.append(it)
        return out

    rec.decisions = _dedupe(rec.decisions, lambda d: d.statement)
    rec.action_items = _dedupe(rec.action_items, lambda a: a.description)
    rec.participants_detected = [p for p in dict.fromkeys(rec.participants_detected) if g.name_ok(p)]
    return rec


def _neighbours(g: "_Grounder", quote: str, label: str) -> set[str]:
    """For a speaker-name evidence quote like 'Thank you, Councillor Miller' said by the NEXT speaker,
    the label that is named is the one that spoke in the paragraph just before the quote."""
    qw = _words(quote)
    if len(qw) < 3:
        return set()
    n = min(4, len(qw))
    grams = _ngrams(qw, n)
    out: set[str] = set()
    for i, tw in enumerate(g._turn_words):
        if grams and sum(1 for x in grams if x in _ngrams(tw, n)) / len(grams) >= 0.7:
            for j in (i - 1, i + 1):
                if 0 <= j < len(g.turns) and g.turns[j][0]:
                    out.add(g.turns[j][0])
    return out


# --------------------------------------------------------------------------------------
# LLM calls
# --------------------------------------------------------------------------------------
def _prepare(transcript: Any) -> tuple[list[tuple[Optional[str], str]], list[str], str]:
    """-> (turns as (speaker, text), lines shown to the LLM, plain text used for grounding)"""
    turns: list[tuple[Optional[str], str]] = []
    if isinstance(transcript, str):
        for block in re.split(r"\n\s*\n|\n", transcript.strip()):
            block = block.strip()
            if not block:
                continue
            m = _LINE_LABEL.match(block)
            turns.append((m.group(1), m.group(2).strip()) if m else (None, block))
    else:
        items = getattr(transcript, "turns", transcript)       # RefinementResult | list[Turn]
        for t in items:
            if isinstance(t, dict):
                turns.append((t.get("speaker"), str(t.get("text", "")).strip()))
            else:
                turns.append((getattr(t, "speaker", None), str(getattr(t, "text", "")).strip()))
    turns = [(sp, tx) for sp, tx in turns if tx]
    if len(turns) == 1 and len(turns[0][1]) > 1500:             # one giant blob: split into sentences
        sp = turns[0][0]
        sents = [x for x in re.split(r"(?<=[.!?])\s+", turns[0][1]) if x]
        turns = [(sp, " ".join(sents[i:i + 4])) for i in range(0, len(sents), 4)]
    lines = [f"{sp}: {tx}" if sp else tx for sp, tx in turns]
    plain = " ".join(tx for _, tx in turns)
    return turns, lines, plain


def _chunk_lines(lines: list[str], limit: int) -> list[list[str]]:
    chunks, cur, size = [], [], 0
    for l in lines:
        if cur and size + len(l) > limit:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(l)
        size += len(l) + 1
    if cur:
        chunks.append(cur)
    return chunks


_COMPUTED_FIELDS = ("quote_verified", "timestamp", "timestamp_s", "agreement_timestamp", "agreement_timestamp_s",
                   "confidence", "confidence_score", "confidence_reasons", "clarity")


def _extract_once(client, model: str, text: str, labelled: bool, part: str = "") -> MeetingRecord:
    system = SYSTEM_PROMPT.replace("__SPEAKER_RULES__", SPEAKER_RULES_LABELLED if labelled else SPEAKER_RULES_PLAIN)
    user = f"{part}TRANSCRIPT:\n\n{text}\n\nReturn the JSON object now."
    schema = MeetingRecord.model_json_schema()
    last_err: Optional[Exception] = None
    for attempt in range(2):
        prompt = user if attempt == 0 else (
            user + f"\n\nYour previous reply did not match the required JSON structure ({last_err}). "
                   "Follow the schema exactly.")
        data = chat_json(client, model, system, prompt, temperature=0.1, max_tokens=4096,
                         schema=schema, schema_name="meeting_record")
        for key in ("warnings", "model", "meeting_duration_minutes", "speaker_labels_used"):   # never trust model for metadata
            data.pop(key, None)
        for coll in ("decisions", "action_items"):          # computed fields are ours, not the model's
            for it in data.get(coll) or []:
                if isinstance(it, dict):
                    for k in _COMPUTED_FIELDS:
                        it.pop(k, None)
        try:
            return MeetingRecord.model_validate(data)
        except ValidationError as e:
            last_err = e
    raise ExtractionError(f"The model returned an invalid meeting record twice: {last_err}")


def _merge_overview(client, model: str, parts: list[MeetingRecord]) -> tuple[str, list[str]]:
    blob = "\n\n".join(f"PART {i + 1} SUMMARY: {p.summary}\nTOPICS: {', '.join(p.key_topics)}"
                       for i, p in enumerate(parts))
    try:
        data = chat_json(client, model, MERGE_PROMPT, blob, temperature=0.1, max_tokens=800)
        topics = [str(t) for t in data.get("key_topics", [])]
        return str(data.get("summary", "")).strip(), list(dict.fromkeys(topics))
    except LLMError:
        return " ".join(p.summary for p in parts), list(dict.fromkeys(t for p in parts for t in p.key_topics))


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def extract_meeting_record(
    transcript: Any,
    duration_s: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
) -> MeetingRecord:
    """transcript: RefinementResult | list[Turn] | plain string (refined transcript)."""
    report = progress_cb or (lambda f, m: None)
    turns, lines, plain = _prepare(transcript)
    if len(plain.strip()) < 20:
        raise ExtractionError("The transcript is too short to generate meeting minutes.")
    labelled = any(sp for sp, _ in turns)

    client = make_llm_client(api_key, base_url)
    model = get_model(model)
    joined = "\n\n".join(lines)

    if len(joined) <= MAX_SINGLE_PASS_CHARS:
        report(0.1, "Generating minutes, decisions and tasks...")
        rec = _extract_once(client, model, joined, labelled)
    else:
        chunks = _chunk_lines(lines, CHUNK_CHARS)
        parts = []
        for i, ch in enumerate(chunks):
            report(0.05 + 0.8 * i / len(chunks), f"Analysing part {i + 1}/{len(chunks)}...")
            parts.append(_extract_once(client, model, "\n\n".join(ch), labelled,
                                       part=f"This is part {i + 1} of {len(chunks)} of one meeting.\n"))
        report(0.9, "Merging parts...")
        summary, topics = _merge_overview(client, model, parts)
        spk: dict[str, SpeakerInfo] = {}
        for p in parts:
            for sp in p.speakers:
                if sp.label not in spk or (spk[sp.label].name == UNSPECIFIED and sp.name != UNSPECIFIED):
                    spk[sp.label] = sp
        rec = MeetingRecord(
            summary=summary, key_topics=topics,
            minutes=[s for p in parts for s in p.minutes],
            decisions=[d for p in parts for d in p.decisions],
            action_items=[a for p in parts for a in p.action_items],
            participants_detected=[n for p in parts for n in p.participants_detected],
            speakers=list(spk.values()),
        )

    rec = ground_record(rec, turns_to_input(turns))
    report(0.82, "Locating evidence in the audio and scoring confidence...")
    rec = score_record(rec, transcript)          # timings come from the original (refined) turns
    rec.model = model
    if duration_s:
        rec.meeting_duration_minutes = round(duration_s / 60.0, 1)
    report(1.0, "Meeting record ready.")
    return rec


def turns_to_input(turns: list[tuple[Optional[str], str]]) -> list[Turn]:
    return [Turn(sp, tx) for sp, tx in turns]


# --------------------------------------------------------------------------------------
# Human-readable rendering (derived from the same object as the JSON)
# --------------------------------------------------------------------------------------
def _cell(t: str) -> str:
    return (t or "").replace("|", "\\|").replace("\n", " ").strip()


def _badge(tier: str, score: int) -> str:
    return f"{TIER_ICON.get(tier, '')} {tier} ({score}%)"


def _top_reasons(reasons: list[str], k: int = 3) -> str:
    """The k most influential reasons (largest point swings), for one-line display."""
    def weight(r: str) -> int:
        m = re.match(r"^([+−])(\d+)", r)
        return int(m.group(2)) if m else (100 if r.startswith("⤓") else 0)
    top = sorted((r for r in reasons if not r.startswith("·")), key=weight, reverse=True)[:k]
    return "; ".join(top)


def render_markdown(rec: MeetingRecord) -> str:
    out = ["# Meeting Minutes", ""]
    meta = []
    if rec.meeting_duration_minutes:
        meta.append(f"**Duration:** ~{rec.meeting_duration_minutes:g} min")
    if rec.participants_detected:
        meta.append("**People mentioned:** " + ", ".join(rec.participants_detected))
    if rec.speaker_labels_used:
        meta.append(f"**Speakers detected:** {len(rec.speakers)}")
    if meta:
        out += [" &nbsp;|&nbsp; ".join(meta), ""]

    out += ["## Summary", "", rec.summary or "_No summary generated._", ""]
    if rec.speaker_labels_used and rec.speakers:
        out += ["## Speakers", "", "| Speaker | Identified as | Evidence |", "|---|---|---|"]
        for sp in rec.speakers:
            ev = f"\"{_cell(sp.evidence)}\"" if sp.evidence else ""
            out.append(f"| {sp.label} | {_cell(sp.name)} | {ev} |")
        out.append("")
    if rec.key_topics:
        out += ["## Key Topics", ""] + [f"- {t}" for t in rec.key_topics] + [""]
    if rec.minutes:
        out += ["## Minutes", ""]
        for s in rec.minutes:
            out += [f"### {s.heading}"] + [f"- {p}" for p in s.points] + [""]

    agreed = [d for d in rec.decisions if d.status == "agreed"]
    proposed = [d for d in rec.decisions if d.status == "proposed"]

    def _decisions(title: str, items: list[Decision], empty: str):
        out.extend([f"## {title}", ""])
        if not items:
            out.extend([f"_{empty}_", ""])
        for i, d in enumerate(items, 1):
            ts = f" `[{d.timestamp}]`" if d.timestamp else ""
            out.append(f"{i}. **{d.statement}**{ts}")
            if d.confidence:
                out.append(f"   - Confidence: {_badge(d.confidence, d.confidence_score)}"
                           + (f" — {_top_reasons(d.confidence_reasons)}" if d.confidence_reasons else ""))
            if d.participants:
                out.append(f"   - Involved: {', '.join(d.participants)}")
            if d.source_quote:
                flag = "" if d.quote_verified else " (⚠ quote could not be verified)"
                out.append(f"   - Evidence: \"{d.source_quote}\"{flag}")
            if d.agreement_quote:
                at = f" `[{d.agreement_timestamp}]`" if d.agreement_timestamp else ""
                out.append(f"   - Settled by: \"{d.agreement_quote}\"{at}")
        if items:
            out.append("")

    _decisions("Key Decisions (agreed)", agreed, "No decisions were clearly agreed in the recording.")
    if proposed:
        _decisions("Proposals / Open Items (not agreed)", proposed, "")

    out += ["## Action Items", ""]
    if not rec.action_items:
        out += ["_No actionable tasks were identified._", ""]
    else:
        out += ["| # | Time | Task | Owner | Deadline | Status | Confidence | Evidence |",
                "|---|------|------|-------|----------|--------|------------|----------|"]
        for i, a in enumerate(rec.action_items, 1):
            ev = f"\"{_cell(a.source_quote)}\"" if a.source_quote else ""
            if a.source_quote and not a.quote_verified:
                ev += " ⚠"
            owner = _cell(a.owner) + (" _(first-person commitment)_" if a.owner_basis == "first_person" else "")
            conf = _badge(a.confidence, a.confidence_score) if a.confidence else ""
            out.append(f"| {i} | {a.timestamp or '-'} | {_cell(a.description)} | {owner} | {_cell(a.deadline)} | "
                       f"{a.status} | {conf} | {ev} |")
        out.append("")

    if rec.decisions or rec.action_items:
        out += ["## Confidence legend", ""]
        out += [f"- {TIER_ICON[t]} **{t}** — {h}" for t, h in TIER_HELP.items()]
        out += ["", "_Scores are computed from the evidence in the recording (verified quotes, explicit agreement or "
                "acceptance, push-back, hedging, speaker turn-taking and speech-recognition clarity). "
                "Timestamps `[mm:ss]` point to where each item is said - listen there to confirm._", ""]

    if rec.warnings:
        out += ["## Verification notes", ""] + [f"- {w}" for w in rec.warnings] + [""]
    out += [f"---\n_Generated by pipeline · LLM: {rec.model}. Owners and deadlines are shown only when stated in the recording; otherwise \"{UNSPECIFIED}\"._"]
    return "\n".join(out)
