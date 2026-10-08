"""
refiner.py  --  STAGE 2: domain-aware transcript refinement  (LLM #1)

Model : Qwen2.5-7B-Instruct via OpenRouter (OpenAI-compatible API),
        model id "qwen/qwen-2.5-7b-instruct". Any other OpenAI-compatible endpoint / model id also works
        (set LLM_MODEL in .env, e.g. to go back to "qwen/qwen-2.5-72b-instruct").
Input : speaker turns / paragraphs from transcriber.build_turns() (no timestamps)
Output: RefinementResult -> refined turns + an audit log of every accepted AND rejected edit

What changed compared with the first version (why it used to make only ONE correction)
--------------------------------------------------------------------------------------
* The old prompt forbade almost everything ("most segments need no change", "when unsure leave
  unchanged") and the model saw 5-word ASR fragments with no domain context.
* Now the model works on whole paragraphs / speaker turns and receives a RETRIEVED context
  (domain_kb.py): a glossary of authoritative terms, "suspect spans" (heard -> likely), spelling
  variants of the same word, and an optional whole-meeting "briefing". It is explicitly asked to
  fix BOTH domain terminology and ordinary ASR grammar/spelling/punctuation slips.
* Safety no longer relies on a timid prompt. Every edit the LLM proposes is diffed against the
  raw text and validated edit-by-edit (validate_refinement): numbers, dates, negation, modals and
  names are protected; rewrites are rejected; terminology edits must be supported by the glossary /
  transcript evidence. Rejected edits are logged with the reason, so nothing is silently lost or
  silently invented.

This file also hosts the small LLM helper (`make_llm_client`, `chat_json`) that extractor.py
re-uses, so both LLM stages share one client configuration.

Env vars (see .env.example):
  OPENROUTER_API_KEY  (or LLM_API_KEY)   LLM_BASE_URL   LLM_MODEL   LLM_JSON_MODE
  optional: OPENROUTER_SITE_URL, OPENROUTER_APP_NAME  (shown on openrouter.ai rankings)
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen-2.5-7b-instruct"

ProgressCB = Optional[Callable[[float, str], None]]


# --------------------------------------------------------------------------------------
# Shared LLM helper (used by refiner AND extractor)
# --------------------------------------------------------------------------------------
class LLMError(Exception):
    """User-presentable failure of an LLM stage."""


class RefinerError(LLMError):
    pass


def make_llm_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    from openai import OpenAI

    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise LLMError(
            "No LLM API key found. Set OPENROUTER_API_KEY in your .env file, "
            "or paste it in the sidebar."
        )
    # Optional attribution headers recommended by OpenRouter (harmless on other providers)
    headers = {"X-OpenRouter-Title": os.getenv("OPENROUTER_APP_NAME", "AI Meeting Assistant")}
    if os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
    return OpenAI(
        api_key=key,
        base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        default_headers=headers,
        timeout=180.0,
        max_retries=0,   # we run our own retry loop below
    )


def get_model(model: Optional[str] = None) -> str:
    return model or os.getenv("LLM_MODEL", DEFAULT_MODEL)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates ``` fences and chatter)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a == -1 or b <= a:
            raise ValueError("no JSON object found in model reply")
        obj = json.loads(t[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("model reply is not a JSON object")
    return obj


def chat_json(
    client,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    schema: Optional[dict] = None,
    schema_name: str = "output",
) -> dict:
    """One chat completion that must return a JSON object. Retries transient errors.

    LLM_JSON_MODE:
      json_object (default) - OpenAI/DashScope JSON mode (prompt must contain the word JSON)
      json_schema           - schema-constrained decoding; use this when self-hosting with vLLM
      none                  - plain text, we just parse the reply
    """
    import openai

    mode = os.getenv("LLM_JSON_MODE", "json_object").lower()
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if mode == "json_schema" and schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    elif mode in ("json_object", "json_schema"):
        kwargs["response_format"] = {"type": "json_object"}

    last: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(**kwargs)
            if not getattr(resp, "choices", None):      # OpenRouter can return an error body with HTTP 200
                err = getattr(resp, "error", None) or "empty response from provider"
                raise ValueError(str(err))                  # treated as retryable below
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMError("The model's reply was cut off (too long). Try a shorter recording.")
            return extract_json(choice.message.content)
        except openai.AuthenticationError:
            raise LLMError("The LLM provider rejected the API key. Check OPENROUTER_API_KEY.")
        except openai.PermissionDeniedError as e:
            raise LLMError(f"The LLM request was rejected (permission denied / blocked): {getattr(e, 'message', e)}")
        except openai.NotFoundError:
            raise LLMError(
                f"Model '{model}' was not found or has no available provider. Check LLM_MODEL "
                "(OpenRouter id: qwen/qwen-2.5-7b-instruct) and your OpenRouter privacy/data settings."
            )
        except openai.BadRequestError as e:
            if "response_format" in kwargs:     # provider doesn't support JSON mode -> degrade
                logger.warning("JSON mode rejected (%s); retrying without response_format", e)
                kwargs.pop("response_format")
                continue
            raise LLMError(f"The LLM request was rejected: {getattr(e, 'message', e)}")
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.APITimeoutError, openai.InternalServerError) as e:
            last = e
            time.sleep(min(3.0 * attempt, 20))
        except ValueError as e:                 # unparsable JSON -> retry once or twice
            last = e
            kwargs["temperature"] = 0.0
        except openai.APIStatusError as e:      # anything else, e.g. OpenRouter 402 = out of credits
            if e.status_code == 402:
                raise LLMError("OpenRouter says you are out of credits (402). Add credits at "
                               "openrouter.ai/credits, or use a ':free' model id.")
            raise LLMError(f"The LLM provider returned an error ({e.status_code}): {getattr(e, 'message', e)}")
    raise LLMError(
        f"The LLM stage failed after several attempts ({type(last).__name__}: {last}). Please retry."
    )




# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
from domain_kb import (  # noqa: E402  (kept below the LLM helper so extractor can import the helper alone)
    GRAMMAR_WORDS, DomainContext, build_context, char_ratio, flat, phonetic,
)
from transcriber import Turn, render_turns  # noqa: E402


@dataclass
class Edit:
    turn: int                       # index of the paragraph / speaker turn
    speaker: Optional[str]
    before: str
    after: str
    kind: str                       # domain term | name consistency | grammar | capitalisation | punctuation | filler | rejected
    accepted: bool
    reason: str                     # why it was accepted / why it was rejected
    context: str = ""               # short raw snippet around the edit


@dataclass
class RefinedTurn:
    speaker: Optional[str]
    raw: str
    text: str
    edits: list[Edit] = field(default_factory=list)
    start: float = 0.0                     # audio timings carried over from the transcriber turn
    end: float = 0.0
    asr_conf: Optional[float] = None       # Whisper clarity of this passage (0..1)

    def as_turn(self, refined: bool = True) -> Turn:
        return Turn(self.speaker, self.text if refined else self.raw, self.start, self.end, self.asr_conf)


@dataclass
class RefinementResult:
    turns: list[RefinedTurn]
    model: str
    domain_context: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)      # suspected terms the LLM left unchanged

    def text(self, timestamps: bool = False) -> str:
        return render_turns([t.as_turn(True) for t in self.turns], timestamps=timestamps)

    def raw_text(self, timestamps: bool = False) -> str:
        return render_turns([t.as_turn(False) for t in self.turns], timestamps=timestamps)

    def as_turns(self) -> list[Turn]:
        return [t.as_turn(True) for t in self.turns]

    @property
    def applied(self) -> list[Edit]:
        return [e for t in self.turns for e in t.edits if e.accepted]

    @property
    def rejected(self) -> list[Edit]:
        return [e for t in self.turns for e in t.edits if not e.accepted]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for e in self.applied:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {"turns": len(self.turns), "edits_applied": len(self.applied),
                "edits_rejected": len(self.rejected), "by_kind": kinds,
                "turns_changed": sum(1 for t in self.turns if t.text != t.raw)}

    def report(self) -> dict:
        def d(e: Edit) -> dict:
            return {"turn": e.turn + 1, "speaker": e.speaker, "before": e.before, "after": e.after,
                    "kind": e.kind, "reason": e.reason, "context": e.context}
        return {"model": self.model, "stats": self.stats(),
                "applied_edits": [d(e) for e in self.applied],
                "rejected_edits": [d(e) for e in self.rejected],
                "suspected_but_unchanged": self.unresolved,
                "domain_context": self.domain_context, "warnings": self.warnings}


# --------------------------------------------------------------------------------------
# Edit-level validator  (deterministic safety net; no LLM involved)
# --------------------------------------------------------------------------------------
_TOK = re.compile(r"\d+(?:[.,:/]\d+)*%?|[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[^\w\s]|_")
_WORD = re.compile(r"[^\W_]", re.UNICODE)
_NEG = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "cannot", "without", "nope", "nowhere"}
_MODALS = {"will", "would", "shall", "should", "must", "can", "cannot", "could", "may", "might"}
_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "half", "quarter", "twice", "double", "triple", "dozen", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "yesterday",
}
_AUX = {"is", "are", "was", "were", "am", "be", "been", "being", "do", "does", "did", "has", "have", "had"}
_NEG_PREFIX = ("dis", "un", "non", "in", "im", "ir", "il", "mis", "anti", "de")
_FILLERS = {"um", "uh", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"}
_DROPPABLE = {"the", "a", "an", "of", "that", "to"}
_SENT_PUNCT = {".", "!", "?"}


@dataclass
class _Tok:
    text: str
    sp: bool           # preceded by whitespace in its own text

    @property
    def low(self) -> str:
        return self.text.lower().replace("’", "'")

    @property
    def is_word(self) -> bool:
        return bool(_WORD.search(self.text))


def _toks(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    prev_end = 0
    for m in _TOK.finditer(text):
        out.append(_Tok(m.group(0), m.start() > prev_end))
        prev_end = m.end()
    return out


def _detok(toks: list[_Tok]) -> str:
    parts: list[str] = []
    for i, t in enumerate(toks):
        if i > 0 and t.sp:
            parts.append(" ")
        parts.append(t.text)
    return "".join(parts)


def _negates(x: str, y: str) -> bool:
    """True when one word is the other with a negating prefix (approve / disapprove, agree / disagree)."""
    if x == y:
        return False
    for a, b in ((x, y), (y, x)):
        for p in _NEG_PREFIX:
            if a == p + b or (a.startswith("dis") and a[3:] == b.rstrip("d")):
                return True
    return False


def _is_neg(low: str) -> bool:
    return low in _NEG or low.endswith("n't")


def _protected_signature(toks: list[_Tok]) -> tuple:
    nums = sorted(t.low for t in toks if any(c.isdigit() for c in t.text) or t.low in _NUMWORDS)
    negs = sorted(t.low for t in toks if _is_neg(t.low))
    mods = sorted(t.low for t in toks if t.low in _MODALS)
    return tuple(nums), tuple(negs), tuple(mods)


def _wtext(toks: list[_Tok]) -> str:
    return " ".join(t.text for t in toks if t.is_word)


def _context_snippet(raw_toks: list[_Tok], i1: int, i2: int, pad: int = 6) -> str:
    return _detok(raw_toks[max(0, i1 - pad): min(len(raw_toks), i2 + pad)]).strip()


def _classify_hunk(O: list[_Tok], N: list[_Tok], raw_toks: list[_Tok], i1: int, raw_flat: set[str],
                   ctx: Optional[DomainContext]) -> tuple[bool, str, str]:
    """Decide on one replace / insert / delete hunk. Returns (accept, kind, reason)."""
    ow, nw = [t for t in O if t.is_word], [t for t in N if t.is_word]
    fo, fn = flat(" ".join(t.text for t in ow)), flat(" ".join(t.text for t in nw))

    # 1) punctuation-only edits are always safe, unless they touch a number
    if not ow and not nw:
        if any(any(c.isdigit() for c in t.text) for t in O + N):
            return False, "rejected", "punctuation edit touches a number"
        return True, "punctuation", "punctuation / sentence boundary"

    # 2) numbers, dates, negations and modals must be identical on both sides
    so, sn = _protected_signature(O), _protected_signature(N)
    if so[0] != sn[0]:
        return False, "rejected", "would change a number / date / quantity"
    if so[1] != sn[1]:
        return False, "rejected", "would change negation"
    if so[2] != sn[2]:
        return False, "rejected", "would change a modal verb (will / should / can ...)"

    # 3) pure deletions
    if not nw:
        low = [t.low for t in ow]
        prev = raw_toks[i1 - 1].low if i1 > 0 else ""
        nxt_i = i1 + len(O)
        nxt = raw_toks[nxt_i].low if nxt_i < len(raw_toks) else ""
        if all(w in _FILLERS for w in low):
            return True, "filler", "removed filler word"
        if len(low) <= 3 and (low == [prev] or low == [nxt] or (len(low) == 2 and low[0] == low[1])):
            return True, "filler", "removed accidental repeat"
        if len(low) <= 2 and all(w in _DROPPABLE for w in low):
            return True, "grammar", "dropped redundant function word"
        return False, "rejected", "would delete content words"

    # 4) pure insertions
    if not ow:
        low = [t.low for t in nw]
        if len(low) <= 2 and all(w in GRAMMAR_WORDS or w in {"is", "are", "was", "were"} for w in low):
            return True, "grammar", "inserted missing function word"
        if ctx is not None and ctx.supports("", fn):
            return True, "domain term", "glossary-backed insertion"
        return False, "rejected", "would insert new content words"

    # 5) replacements
    if ctx is not None and ctx.supports(fo, fn):
        if len(ow) <= 6 and len(nw) <= 6:
            return True, "domain term", "backed by glossary / dictionary / transcript evidence"
    if len(ow) > 8 or len(nw) > 10:
        return False, "rejected", "rewrites too many words"

    # names: do not turn one person into another
    def cap(ts: list[_Tok], base: int) -> bool:
        for k, t in enumerate(ts):
            if t.is_word and t.text[:1].isupper() and not t.text.isupper():
                idx = base + k
                after_stop = idx == 0 or (idx > 0 and raw_toks[idx - 1].text in _SENT_PUNCT)
                if not after_stop:
                    return True
        return False

    name_like = cap(O, i1) or any(t.text[:1].isupper() and not t.text.isupper() for t in nw)
    sim = char_ratio(fo.replace(" ", ""), fn.replace(" ", ""))
    ph = char_ratio(phonetic(fo), phonetic(fn)) if phonetic(fo) and phonetic(fn) else 0.0
    if name_like:
        # accept only a respelling of the same name into a form that already occurs in the raw transcript
        if all(w in raw_flat for w in fn.split()) and (sim >= 0.6 or ph >= 0.8):
            return True, "name consistency", "same name spelled differently elsewhere in the transcript"
        return False, "rejected", "would change a proper name without evidence"

    # grammar / spelling slip: small, clearly-similar words, or a function-word swap
    if len(ow) <= 3 and len(nw) <= 3:
        a, b = [t.low for t in ow], [t.low for t in nw]
        # "the there" -> "their": allow dropping ONE leading/trailing article-like word
        if len(a) == len(b) + 1 and (a[0] in GRAMMAR_WORDS or a[-1] in GRAMMAR_WORDS):
            a = a[1:] if a[0] in GRAMMAR_WORDS else a[:-1]
        elif len(b) == len(a) + 1 and (b[0] in GRAMMAR_WORDS or b[-1] in GRAMMAR_WORDS):
            b = b[1:] if b[0] in GRAMMAR_WORDS else b[:-1]
        if len(a) == len(b):
            if all(x in GRAMMAR_WORDS or x in _AUX for x in a + b):
                return True, "grammar", "function-word fix"
            if any(_negates(x, y) for x, y in zip(a, b)):
                return False, "rejected", "would flip the meaning of a word (e.g. approve / disapprove)"
            pair_sim = min(char_ratio(x, y) for x, y in zip(a, b))
            if pair_sim >= 0.75 or (ph >= 0.9 and pair_sim >= 0.6):
                return True, "grammar", f"spelling / word-form fix (similarity {pair_sim:.2f})"
    return False, "rejected", "replaces words with unrelated words and has no supporting evidence"


def validate_refinement(raw: str, proposed: str, ctx: Optional[DomainContext] = None,
                        turn: int = 0, speaker: Optional[str] = None,
                        known_words: Optional[set] = None) -> tuple[str, list[Edit]]:
    """Diff `proposed` (LLM output) against `raw` and keep only the safe, supported edits.
    Returns (final_text, edit_log)."""
    R, P = _toks(raw), _toks(proposed)
    if not P or not R:
        return raw, []
    rw, pw = sum(t.is_word for t in R), sum(t.is_word for t in P)
    if rw and not (0.5 <= pw / rw <= 1.6):
        return raw, [Edit(turn, speaker, raw[:60] + "...", proposed[:60] + "...", "rejected", False,
                          "whole passage rewritten (length changed too much); kept raw text")]

    raw_flat = set(flat(" ".join(t.text for t in R if t.is_word)).split()) | (known_words or set())
    sm = difflib.SequenceMatcher(None, [t.low for t in R], [t.low for t in P], autojunk=False)
    out: list[_Tok] = []
    edits: list[Edit] = []

    def log(O, N, i1, i2, kind, ok, why):
        edits.append(Edit(turn, speaker, _detok(O).strip() or "(nothing)", _detok(N).strip() or "(deleted)",
                          kind, ok, why, _context_snippet(R, i1, i2)))

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        O, N = R[i1:i2], P[j1:j2]
        if op == "equal":
            for k, (r, p) in enumerate(zip(O, N)):
                if r.text == p.text:
                    out.append(r)
                    continue
                idx = i1 + k
                sentence_start = idx == 0 or R[idx - 1].text in _SENT_PUNCT
                lowering = r.text[:1].isupper() and p.text[:1].islower() and not sentence_start
                if lowering or (r.text.isupper() and len(r.text) > 1 and not p.text.isupper()):
                    out.append(r)           # never lower-case a name / acronym
                    log([r], [p], idx, idx + 1, "rejected", False, "would lower-case a name or acronym")
                else:
                    out.append(_Tok(p.text, r.sp))
                    log([r], [p], idx, idx + 1, "capitalisation", True, "capitalisation")
            continue
        ok, kind, why = _classify_hunk(O, N, R, i1, raw_flat, ctx)
        log(O, N, i1, i2, kind, ok, why)
        if ok:
            new = [_Tok(t.text, t.sp) for t in N]
            if new and new[0].is_word:
                new[0].sp = O[0].sp if O else True
            out.extend(new)
        else:
            out.extend(O)
    final = _detok(out)
    # keep the original paragraph's outer whitespace style
    return final.strip(), edits


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert editor of speech-recognition (ASR) transcripts of real meetings. The transcript was produced by \
Whisper, which mishears specialised vocabulary and often produces sloppy grammar, wrong homophones and missing \
punctuation. Your job: return each passage exactly as the speaker INTENDED it - corrected, readable, and equal in \
meaning to what was said.

Fix BOTH kinds of problem:
1. DOMAIN TERMINOLOGY - mis-heard technical terms, acronyms, job titles, organisations, place names and jargon. \
Words that are phonetically close to a glossary term but do not make sense in the sentence are almost certainly \
recognition errors (example: "we deployed it on cooper netties" -> "Kubernetes"; "our sequel database" -> "SQL \
database"; "the CEO is on a course" in a council meeting where a "CAO" is introduced -> "CAO"). Use the GLOSSARY and \
SUSPECT SPANS you are given, the surrounding passages, and your own knowledge of the domain. A suspect span is a \
hint, not an order: apply it only when the sentence really makes sense with the substitution.
2. ORDINARY ASR ERRORS - wrong word forms and agreement, wrong homophones (their/there, to/too, form/forum, \
counsel/council), missing or wrong small words (articles, prepositions, auxiliaries), run-on text without \
capitals or sentence punctuation, obvious spelling slips. Also drop pure fillers (um, uh) and accidental stutters \
("the the").
3. NAMES - if the same person / place is spelled several different ways, use ONE spelling (the glossary spelling, \
else the one that appears most often or in the most official-looking context).

NEVER do any of the following:
- change, add or remove a number, date, time, amount, percentage or unit;
- add, remove or move a negation (not, no, never, n't) or change a modal (will / would / should / can / may ...);
- change who does what, who agrees or objects, or turn a proposal into a decision (or the reverse);
- invent a name, figure, owner or deadline that is not in the passage; replace a person's name with another person;
- summarise, shorten, reorder, translate, add commentary, or restyle correct wording (no synonym swaps);
- merge or split passages, or change passage ids.
If a span is genuinely ambiguous, leave it as it is.

You get numbered passages (the speaker label is only context; do not include it in the text). Return ONLY a JSON \
object of the form {"passages": [{"id": <number>, "text": "<corrected passage>"}]} that contains EVERY passage id \
you were given, in order.
"""

BRIEFING_SYSTEM = """\
You prepare a briefing for a transcript editor. Read the whole ASR transcript of a meeting and report what the \
meeting is about and which words were probably mis-recognised. Return ONLY JSON:
{"domain": "<short description>", "meeting_type": "<e.g. rural-municipality council meeting>",
 "organizations": [..], "people": [..], "places": [..], "key_terms": [..],
 "suspected_misrecognitions": [{"heard": "<exact words as they appear in the transcript>", "likely": "<correct form>", "why": "<very short reason>"}]}
Rules: "heard" must be copied VERBATIM from the transcript. Only list a misrecognition when the heard words make \
little sense in context and the likely form is a standard term, title or name that the transcript itself supports \
(for example, a role that is introduced elsewhere). Do not include numbers or dates. At most 30 items per list.
"""


def _sample_for_briefing(text: str, limit: int = 28000) -> str:
    if len(text) <= limit:
        return text
    third = limit // 3
    mid = len(text) // 2
    return text[:third] + "\n[...]\n" + text[mid - third // 2: mid + third // 2] + "\n[...]\n" + text[-third:]


def make_briefing(client, model: str, text: str) -> Optional[dict]:
    try:
        return chat_json(client, model, BRIEFING_SYSTEM, "TRANSCRIPT:\n" + _sample_for_briefing(text),
                         temperature=0.0, max_tokens=2500)
    except LLMError as e:
        logger.warning("briefing skipped: %s", e)
        return None


def _batches(turns: list[Turn], max_chars: int = 4200, max_items: int = 10):
    cur: list[int] = []
    size = 0
    for i, t in enumerate(turns):
        if cur and (size + len(t.text) > max_chars or len(cur) >= max_items):
            yield cur
            cur, size = [], 0
        cur.append(i)
        size += len(t.text)
    if cur:
        yield cur


def _user_message(turns: list[Turn], idxs: list[int], ctx: DomainContext) -> str:
    batch_text = " ".join(turns[i].text for i in idxs)
    parts: list[str] = []
    block = ctx.batch_block(batch_text)
    if block:
        parts.append(block)
        parts.append("")
    first = idxs[0]
    if first > 0:
        prev = turns[first - 1]
        parts.append(f"PRECEDING PASSAGE (context only - do not return it): {prev.text[-500:]}")
        parts.append("")
    parts.append("PASSAGES TO CORRECT:")
    for i in idxs:
        t = turns[i]
        label = f" (speaker: {t.speaker})" if t.speaker else ""
        parts.append(f"### id={i}{label}\n{t.text}")
    parts.append("\nReturn the JSON object now.")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def refine_transcript(
    turns: list[Turn],
    domain_hints: str = "",
    use_dictionaries: bool = True,
    use_keybert: bool = True,
    use_briefing: bool = True,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
    base_url: Optional[str] = None,
) -> RefinementResult:
    """Refine speaker turns. `domain_hints` is the optional free-text box in the UI
    (comma/newline separated terms, 'alias -> Term' lines, or plain notes)."""
    report = progress_cb or (lambda f, m: None)
    if not turns or not any(t.text.strip() for t in turns):
        raise RefinerError("There is no transcript text to refine.")
    model_id = get_model(model)
    client = make_llm_client(api_key, base_url)

    raw_all = "\n\n".join(t.text for t in turns)
    report(0.03, "Retrieving domain context (dictionaries, keyphrases)...")
    ctx = build_context(raw_all, user_hints=domain_hints, use_dictionaries=use_dictionaries, use_keybert=use_keybert)

    warnings: list[str] = list(ctx.notes)
    if use_briefing:
        report(0.10, "Reading the whole meeting to build a domain briefing...")
        brief = make_briefing(client, model_id, raw_all)
        if brief:
            ctx.add_briefing(brief, raw_all)
        else:
            warnings.append("The LLM briefing step failed and was skipped (dictionary + transcript evidence still used).")

    system = SYSTEM_PROMPT
    gloss = ctx.glossary_block()
    if gloss:
        system += "\n\n" + gloss

    refined: list[Optional[str]] = [None] * len(turns)
    batches = list(_batches(turns))
    for bi, idxs in enumerate(batches):
        report(0.15 + 0.8 * bi / max(1, len(batches)), f"Refining passages {idxs[0] + 1}-{idxs[-1] + 1} of {len(turns)}...")
        try:
            data = chat_json(client, model_id, system, _user_message(turns, idxs, ctx),
                             temperature=0.1, max_tokens=4096)
        except LLMError as e:
            if "cut off" in str(e) and len(idxs) > 1:      # retry the batch one passage at a time
                data = {"passages": []}
                for i in idxs:
                    try:
                        d1 = chat_json(client, model_id, system, _user_message(turns, [i], ctx),
                                       temperature=0.1, max_tokens=4096)
                        data["passages"].extend(d1.get("passages") or [])
                    except LLMError as e2:
                        warnings.append(f"Passage {i + 1} left unrefined: {e2}")
            else:
                raise RefinerError(str(e))
        items = data.get("passages") or data.get("turns") or data.get("segments") or []
        if isinstance(items, dict):
            items = [{"id": k, "text": v} for k, v in items.items()]
        for it in items:
            try:
                i = int(it.get("id"))
                txt = str(it.get("text") or "").strip()
            except (TypeError, ValueError, AttributeError):
                continue
            if i in idxs and txt:
                refined[i] = txt
        missing = [i for i in idxs if refined[i] is None]
        if missing:
            warnings.append(f"The model did not return passage(s) {', '.join(str(m + 1) for m in missing)}; raw text kept.")

    report(0.96, "Validating every edit against the raw transcript...")
    known = set(flat(raw_all).split())
    out_turns: list[RefinedTurn] = []
    for i, t in enumerate(turns):
        proposed = refined[i]
        timing = dict(start=getattr(t, "start", 0.0) or 0.0, end=getattr(t, "end", 0.0) or 0.0,
                      asr_conf=getattr(t, "asr_conf", None))
        if not proposed:
            out_turns.append(RefinedTurn(t.speaker, t.text, t.text, **timing))
            continue
        final, edits = validate_refinement(t.text, proposed, ctx, turn=i, speaker=t.speaker, known_words=known)
        out_turns.append(RefinedTurn(t.speaker, t.text, final, edits, **timing))

    # suspected terms that survive in the final text (so the user can see what was NOT fixed)
    final_flat = " " + flat(" ".join(t.text for t in out_turns)) + " "
    unresolved = []
    for h in ctx.hints:
        if h.score >= 0.95 and f" {h.key()} " in final_flat:
            unresolved.append({"heard": h.span, "likely": h.suggestion, "note": h.note, "source": h.source})
    report(1.0, "Refinement complete.")
    return RefinementResult(turns=out_turns, model=model_id, domain_context=ctx.to_dict(),
                            warnings=warnings, unresolved=unresolved)


if __name__ == "__main__":
    import sys
    print("This module is used by app.py. Run:  streamlit run app.py")
    sys.exit(0)
