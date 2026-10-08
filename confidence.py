"""
confidence.py  --  STAGE 4b: evidence localisation + confidence calibration  (deterministic, no LLM)

Why a separate stage?
---------------------
LLM #2 (extractor.py) decides WHAT the decisions / tasks are. It is a poor judge of how sure we
should be about each one, because LLM self-reported confidence is badly calibrated. So the work is split:

  * LLM #2 only reports *evidence features* it can read off the text
        decisions    : agreement_signal (explicit / implicit / none), objection (none / resolved /
                       unresolved), hedged (bool)
        action items : acceptance (explicit / implicit / none), hedged (bool)
  * this module turns those features + hard evidence from the pipeline into a transparent score:
        - is the evidence quote really in the transcript, and WHERE (-> audio timestamp)?
        - lexical cues in the quote and the next few turns (formal settlement words, assent,
          push-back, hedges, first-person commitments) - used to cross-check the LLM features
        - did a *different* speaker acknowledge it? (needs diarization)
        - ASR clarity of the passage (Whisper avg_logprob, from transcriber.py)
        - did the refiner have to change words inside the quote? (from refiner.py's audit log)

Every point added or removed is recorded as a human-readable reason, so the user sees WHY an item
is "High chance" rather than "Confirmed", and can press the timestamp to listen for themselves.

Tiers (score 0-100)
-------------------
  Confirmed    >= 80   explicitly settled / accepted, verified quote, clear audio
  High chance  60-79   agreed or accepted, but implicitly, or one weaker signal
  Ambiguous    40-59   mixed signals: push-back, hedging, unclear audio, or a well-supported proposal
  Low chance   <  40   floated / suggested only, weak or missing evidence

Hard rules (so the score can never contradict the PS rules):
  * a "proposed" decision / action item is capped at 55 -> never above Ambiguous
  * "Confirmed" requires explicit settlement (decisions) or explicit acceptance by a stated owner (tasks)
  * an agreed decision with an unresolved objection is forced into Ambiguous
  * unclear audio caps an item at High chance (listen to confirm)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from transcriber import fmt_ts

TIERS = ("Confirmed", "High chance", "Ambiguous", "Low chance")
TIER_ICON = {"Confirmed": "🟢", "High chance": "🔵", "Ambiguous": "🟠", "Low chance": "🔴"}
TIER_HELP = {
    "Confirmed": "explicitly settled / accepted in the recording, quote verified, audio clear",
    "High chance": "agreed or accepted, but implicitly or with one weaker signal",
    "Ambiguous": "mixed signals (push-back, hedging, unclear audio) or a well-supported proposal",
    "Low chance": "only suggested or weakly evidenced - treat as an open idea",
}
SEEK_LEAD_S = 2.0          # start playback a little before the quote so the user hears the lead-in

_MINOR_EDIT_KINDS = {"capitalisation", "punctuation", "filler"}
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("’", "'").replace("'", ""))


def _norm(text: str) -> str:
    return " " + " ".join(_words(text)) + " "


def _ngrams(ws: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


# --------------------------------------------------------------------------------------
# Lexical cue lists (matched on normalised text: lower-case, apostrophes removed -> "I'll" = "ill")
# --------------------------------------------------------------------------------------
FORMAL_SETTLE = [
    "motion carried", "so carried", "carried unanimously", "is carried", "unanimous", "unanimously",
    "motion passes", "motion passed", "all in favour", "all in favor", "so decided", "so resolved",
    "is adopted", "is approved", "are approved", "we have decided", "weve decided", "its decided",
    "it is decided", "final decision", "decision is", "thats final", "that is final", "lets lock",
    "locked in", "signed off", "sign off on", "we are going with", "were going with", "we will go with",
    "well go with", "thats settled", "that is settled", "agreed then", "thats agreed", "that is agreed",
]
STRONG_AGREE = [
    "agreed", "i agree", "we agree", "sounds good", "that works", "works for me", "lets go with",
    "lets do it", "lets do that", "go ahead", "makes sense", "no objection", "no objections",
    "fair enough", "absolutely", "definitely", "im on board", "on board", "deal", "perfect",
    "confirmed", "consensus", "settled", "approved", "fine by me", "happy with that",
]
WEAK_AGREE = ["yes", "yeah", "yep", "okay", "ok", "sure", "right", "fine", "great", "alright", "cool"]
OBJECTION = [
    "disagree", "dont agree", "do not agree", "not sure", "not convinced", "im not convinced",
    "concern", "concerned", "worried", "hold on", "wait", "i dont think", "dont think we", "but what about",
    "problem with", "i object", "objection", "against it", "against that", "rather not", "no way",
    "push back", "too risky", "cant do", "cannot do", "not a good idea", "bad idea", "i doubt",
    "not yet", "revisit", "table this", "park this", "lets not", "we shouldnt",
]
NEGATED_OBJECTION = ["no objection", "no objections", "no concerns", "no concern", "not against",
                     "dont disagree", "no problem with", "without objection"]
HEDGE = [
    "maybe", "perhaps", "probably", "might", "possibly", "i think", "i guess", "i suppose", "what if",
    "should we", "could we", "we could", "consider", "kind of", "sort of", "tentatively", "for now",
    "potentially", "ideally", "hopefully", "try to", "if possible", "if we can", "depends", "it depends",
    "lets see", "we will see", "well see", "not sure", "in principle", "pending",
]
COMMIT = [
    "ill", "i will", "i can take", "ill take", "let me", "im on it", "on it", "ill handle", "will do",
    "ill do", "ill send", "i shall", "i promise", "consider it done", "i got it", "ive got it",
    "i can do", "sure ill", "yes ill", "ill get", "ill make sure", "i will make sure", "ill own",
    "leave it with me", "i can handle", "count me in", "i volunteer",
]


def _hits(norm: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if f" {p} " in norm]


def _objections(norm: str) -> list[str]:
    for p in NEGATED_OBJECTION:                          # "no objections" is agreement, not push-back
        norm = norm.replace(f" {p} ", " ")
    return _hits(norm, OBJECTION)


# --------------------------------------------------------------------------------------
# Transcript view with timings
# --------------------------------------------------------------------------------------
@dataclass
class TurnMeta:
    speaker: Optional[str]
    text: str
    start: float = 0.0
    end: float = 0.0
    asr_conf: Optional[float] = None
    edited_words: set = field(default_factory=set)       # words introduced by substantive refiner edits
    words: list = field(default_factory=list)

    def __post_init__(self):
        self.words = _words(self.text)


_LINE_LABEL = re.compile(r"^(Speaker \d+):\s*(.*)$")


def build_turn_meta(transcript: Any) -> list[TurnMeta]:
    """RefinementResult | list[Turn | RefinedTurn | dict] | str  ->  [TurnMeta]."""
    out: list[TurnMeta] = []
    if isinstance(transcript, str):
        for block in re.split(r"\n\s*\n|\n", transcript.strip()):
            block = block.strip()
            if block:
                m = _LINE_LABEL.match(block)
                out.append(TurnMeta(m.group(1), m.group(2)) if m else TurnMeta(None, block))
        return out
    for t in getattr(transcript, "turns", transcript):
        get = (lambda k, d=None: t.get(k, d)) if isinstance(t, dict) else (lambda k, d=None: getattr(t, k, d))
        text = str(get("text", "") or "").strip()
        if not text:
            continue
        edited: set[str] = set()
        for e in get("edits", []) or []:
            if getattr(e, "accepted", False) and getattr(e, "kind", "") not in _MINOR_EDIT_KINDS:
                edited.update(_words(getattr(e, "after", "")))
        out.append(TurnMeta(get("speaker"), text, float(get("start", 0.0) or 0.0),
                            float(get("end", 0.0) or 0.0), get("asr_conf"), edited))
    return out


@dataclass
class Hit:
    turn: int
    time_s: Optional[float]
    ratio: float


class EvidenceLocator:
    def __init__(self, turns: list[TurnMeta]):
        self.turns = turns
        self.has_times = any(t.end > 0 for t in turns)
        self._grams_cache: dict[tuple[int, int], set] = {}

    def _tgrams(self, i: int, n: int) -> set:
        k = (i, n)
        if k not in self._grams_cache:
            self._grams_cache[k] = _ngrams(self.turns[i].words, n)
        return self._grams_cache[k]

    def locate(self, quote: str) -> Optional[Hit]:
        """Find the paragraph that contains (most of) `quote` and estimate the time the quote starts
        by interpolating the word position inside that paragraph's [start, end] audio span."""
        qw = _words(quote)
        if len(qw) < 3 or not self.turns:
            return None
        n = min(4, len(qw))
        grams = _ngrams(qw, n)
        best: Optional[tuple[float, int]] = None
        for i in range(len(self.turns)):
            tg = self._tgrams(i, n)
            if not tg:
                continue
            r = sum(1 for g in grams if g in tg) / len(grams)
            if r > 0 and (best is None or r > best[0]):
                best = (r, i)
        if best is None or best[0] < 0.4:
            return None
        ratio, i = best
        t = self.turns[i]
        pos = 0
        for k in range(len(t.words) - n + 1):
            if tuple(t.words[k:k + n]) in grams:
                pos = k
                break
        time_s = None
        if self.has_times and t.end > t.start:
            time_s = round(t.start + (t.end - t.start) * pos / max(1, len(t.words)), 1)
        elif self.has_times:
            time_s = round(t.start, 1)
        return Hit(i, time_s, ratio)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
class _Score:
    def __init__(self, base: int = 50):
        self.value = base
        self.reasons: list[str] = []

    def add(self, pts: int, why: str):
        if pts == 0:
            self.reasons.append(f"· {why}")
            return
        self.value += pts
        self.reasons.append(f"{'+' if pts > 0 else '−'}{abs(pts)} {why}")

    def cap(self, hi: int, why: str):
        if self.value > hi:
            self.value = hi
            self.reasons.append(f"⤓ capped at {hi}: {why}")

    def floor(self, lo: int, why: str):
        if self.value < lo:
            self.value = lo
            self.reasons.append(f"⤒ raised to {lo}: {why}")

    def final(self) -> int:
        return int(max(0, min(97, round(self.value))))      # never claim 100% - the user should still be able to verify


def tier_of(score: int) -> str:
    if score >= 80:
        return "Confirmed"
    if score >= 60:
        return "High chance"
    if score >= 40:
        return "Ambiguous"
    return "Low chance"


def _clarity(turns: list[TurnMeta], idxs: list[int]) -> Optional[float]:
    vals = [turns[i].asr_conf for i in idxs if turns[i].asr_conf is not None]
    return round(min(vals), 2) if vals else None


def _apply_clarity(sc: _Score, clarity: Optional[float]) -> None:
    if clarity is None:
        return
    if clarity < 0.5:
        sc.add(-15, f"speech recognition was unsure in this passage (clarity {clarity:.2f}) - listen to confirm")
        sc.cap(79, "audio unclear")
    elif clarity < 0.65:
        sc.add(-6, f"passage only moderately clear (clarity {clarity:.2f})")
    else:
        sc.add(0, f"audio clear in this passage (clarity {clarity:.2f})")


def _refiner_touched(turns: list[TurnMeta], hit: Optional[Hit], quote: str) -> list[str]:
    if hit is None:
        return []
    return sorted(set(_words(quote)) & turns[hit.turn].edited_words)


def _window(turns: list[TurnMeta], start: int, until: Optional[int], extra: int = 2, cap: int = 4) -> list[int]:
    end = max(start + extra, until if until is not None else start)
    end = min(end, start + cap, len(turns) - 1)
    return list(range(start + 1, end + 1))


def score_decision(d: Any, turns: list[TurnMeta], loc: EvidenceLocator) -> None:
    sc = _Score(50)
    src = loc.locate(d.source_quote) if d.source_quote else None
    agr = loc.locate(d.agreement_quote) if d.agreement_quote else None
    src_ok = bool(getattr(d, "quote_verified", False)) and src is not None
    agr_ok = bool(d.agreement_quote) and agr is not None

    # ---- where is it in the audio?
    anchor = src or agr
    d.timestamp_s = max(0.0, anchor.time_s - SEEK_LEAD_S) if anchor and anchor.time_s is not None else None
    d.timestamp = fmt_ts(anchor.time_s) if anchor and anchor.time_s is not None else ""
    d.agreement_timestamp_s = max(0.0, agr.time_s - SEEK_LEAD_S) if agr and agr.time_s is not None else None
    d.agreement_timestamp = fmt_ts(agr.time_s) if agr and agr.time_s is not None else ""

    # ---- evidence present?
    if src_ok:
        sc.add(5, "statement quote found verbatim in the transcript")
    elif agr_ok:
        sc.add(-5, "statement quote not found, but the agreement quote is")
    else:
        sc.add(-25, "no verifiable quote in the transcript")

    # ---- lexical evidence around the decision
    idx = (src or agr).turn if (src or agr) else None
    win = _window(turns, idx, agr.turn if agr else None) if idx is not None else []
    # only VERIFIED quotes may contribute wording cues (an invented quote proves nothing)
    quote_norm = _norm((d.source_quote if src_ok else "") + " " + (d.agreement_quote if agr_ok else ""))
    after_norm = _norm(" ".join(turns[i].text for i in win))
    formal = _hits(quote_norm, FORMAL_SETTLE) or (_hits(after_norm, FORMAL_SETTLE) if win else [])
    strong = _hits(quote_norm, STRONG_AGREE) or _hits(after_norm, STRONG_AGREE)
    weak = _hits(after_norm, WEAK_AGREE)
    objections = _objections(after_norm) + (_objections(_norm(d.source_quote)) if src_ok else [])
    hedges = _hits(_norm(d.source_quote), HEDGE)

    # ---- agreement signal: LLM feature, cross-checked against the text
    sig = getattr(d, "agreement_signal", "none")
    if sig == "explicit" and not (agr_ok or formal or strong):
        sig = "implicit"
        sc.add(0, "model said 'explicitly agreed' but no agreement words were found nearby -> treated as implicit")
    if sig == "none" and (formal or (agr_ok and strong)):
        sig = "implicit"
        sc.add(0, "agreement words found nearby although the model did not flag them")
    if not (src_ok or agr_ok) and sig != "none":
        sig = "none"
        sc.add(0, "agreement claimed by the model, but nothing in the transcript backs it up")
    if sig == "explicit":
        sc.add(25, "explicit agreement stated")
    elif sig == "implicit":
        sc.add(10, "implicit consensus (assent / nobody objected)")
    else:
        sc.add(-15, "no sign that the group agreed")

    if agr_ok:
        sc.add(12, f"settlement quote verified ({d.agreement_timestamp or 'time n/a'})")
    if formal:
        sc.add(10, f"formal settlement wording: “{formal[0]}”")
    elif strong:
        sc.add(5, f"assent wording: “{strong[0]}”")
    elif weak and sig != "none":
        sc.add(2, f"mild assent nearby: “{weak[0]}”")

    # ---- did someone else acknowledge it?
    if src and agr and turns[src.turn].speaker and turns[agr.turn].speaker \
            and turns[src.turn].speaker != turns[agr.turn].speaker:
        sc.add(5, f"acknowledged by a different speaker ({turns[agr.turn].speaker})")

    # ---- push-back and hedging
    obj = getattr(d, "objection", "none")
    if obj == "unresolved":
        sc.add(-25, "an objection was raised and not resolved")
    elif obj == "resolved":
        sc.add(-5, "an objection was raised but then resolved")
    elif objections:
        sc.add(-8, f"possible push-back nearby: “{objections[0]}”")
    if getattr(d, "hedged", False) or hedges:
        pen = min(20, 8 * max(1, len(hedges)))
        sc.add(-pen, "tentative wording" + (f": “{', '.join(hedges[:3])}”" if hedges else ""))

    # ---- transcript quality
    touched = _refiner_touched(turns, src, d.source_quote)
    if touched:
        sc.add(-3, f"refiner corrected words inside the quote ({', '.join(touched[:3])}) - check the audio")
    _apply_clarity(sc, _clarity(turns, [h.turn for h in (src, agr) if h]))
    d.clarity = _clarity(turns, [h.turn for h in (src, agr) if h])

    # ---- hard rules
    if not (src_ok or agr_ok):
        sc.cap(30, "the evidence could not be found in the recording")
    if d.status == "proposed":
        sc.cap(55, "it is a proposal, not an agreed decision")
    else:
        if not (sig == "explicit" and (agr_ok or formal)):
            sc.cap(79, "'Confirmed' needs an explicit, quotable settlement")
        if obj == "unresolved":
            sc.cap(59, "agreed, yet an objection is still open")
        if src_ok or agr_ok:
            sc.floor(40, "the transcript was classified as agreed - kept visible as Ambiguous")

    d.confidence_score = sc.final()
    d.confidence = tier_of(d.confidence_score)
    d.confidence_reasons = sc.reasons


def score_action(a: Any, turns: list[TurnMeta], loc: EvidenceLocator, unspecified: str = "Unspecified") -> None:
    sc = _Score(50)
    hit = loc.locate(a.source_quote) if a.source_quote else None
    a.timestamp_s = max(0.0, hit.time_s - SEEK_LEAD_S) if hit and hit.time_s is not None else None
    a.timestamp = fmt_ts(hit.time_s) if hit and hit.time_s is not None else ""
    ok = bool(getattr(a, "quote_verified", False)) and hit is not None

    if ok:
        sc.add(5, "task quote found verbatim in the transcript")
    else:
        sc.add(-25, "no verifiable quote in the transcript")

    if a.status == "assigned":
        sc.add(15, "explicitly assigned / taken on")
    elif a.status == "proposed":
        sc.add(-15, "only suggested - nobody committed")
    else:
        sc.add(0, "needed work, but no owner was stated")

    qn = _norm(a.source_quote) if ok else " "
    win = _window(turns, hit.turn, None, extra=1, cap=2) if hit else []
    reply_norm = _norm(" ".join(turns[i].text for i in win))
    commits = _hits(qn, COMMIT)
    replies = _hits(reply_norm, COMMIT) + _hits(reply_norm, STRONG_AGREE)
    acc = getattr(a, "acceptance", "none")
    if acc == "explicit" and not (commits or replies or a.owner_basis == "first_person"):
        acc = "implicit"
        sc.add(0, "model said 'explicitly accepted' but no acceptance words were found -> treated as implicit")
    if acc == "none" and commits and a.owner != unspecified:
        acc = "implicit"
    if acc == "explicit":
        sc.add(20, "owner clearly accepted the task")
    elif acc == "implicit":
        sc.add(8, "assigned and not refused, but no clear acknowledgement")
    else:
        sc.add(-10, "nobody accepted the task")
    if commits:
        sc.add(5, f"commitment wording: “{commits[0]}”")
    elif replies and a.owner != unspecified:
        sc.add(3, f"acknowledged in the reply: “{replies[0]}”")

    if a.owner != unspecified:
        sc.add(5, f"owner stated ({'first-person' if a.owner_basis == 'first_person' else 'named'})")
    if a.deadline != unspecified:
        sc.add(5, f"deadline stated: “{a.deadline}”")

    hedges = _hits(qn, HEDGE)
    if getattr(a, "hedged", False) or hedges:
        sc.add(-min(20, 8 * max(1, len(hedges))),
               "tentative wording" + (f": “{', '.join(hedges[:3])}”" if hedges else ""))
    objections = _objections(reply_norm)
    if objections:
        sc.add(-8, f"possible push-back right after: “{objections[0]}”")

    touched = _refiner_touched(turns, hit, a.source_quote)
    if touched:
        sc.add(-3, f"refiner corrected words inside the quote ({', '.join(touched[:3])})")
    a.clarity = _clarity(turns, [hit.turn] if hit else [])
    _apply_clarity(sc, a.clarity)

    if not ok:
        sc.cap(30, "the evidence could not be found in the recording")
    if a.status == "proposed":
        sc.cap(55, "a suggestion, not a confirmed task")
    if not (a.status == "assigned" and acc == "explicit" and ok and a.owner != unspecified):
        sc.cap(79, "'Confirmed' needs a stated owner who explicitly accepted")

    a.confidence_score = sc.final()
    a.confidence = tier_of(a.confidence_score)
    a.confidence_reasons = sc.reasons


def score_record(rec: Any, transcript: Any) -> Any:
    """Locate every decision / action item in the audio and attach a calibrated confidence tier.
    Mutates and returns `rec` (an extractor.MeetingRecord)."""
    turns = build_turn_meta(transcript)
    loc = EvidenceLocator(turns)
    for d in rec.decisions:
        score_decision(d, turns, loc)
    for a in rec.action_items:
        score_action(a, turns, loc)
    order = {t: i for i, t in enumerate(TIERS)}
    # agreed first, then by confidence, then chronologically
    rec.decisions.sort(key=lambda d: (d.status != "agreed", order.get(d.confidence, 9),
                                      d.timestamp_s if d.timestamp_s is not None else 1e9))
    rec.action_items.sort(key=lambda a: a.timestamp_s if a.timestamp_s is not None else 1e9)
    if not loc.has_times:
        rec.warnings.append("No audio timings were available, so timestamps could not be attached.")
    return rec
