"""
verification.py -- transcript-grounded Chain-of-Verification (CoVe)

Adapted for the meeting-assistant pipeline:
  1) take the draft meeting record produced by LLM #2
  2) create atomic claims from the draft
  3) retrieve transcript evidence for each claim
  4) verify each claim independently (the verifier does NOT receive the draft record)
  5) cross-check the result and conservatively revise the record

The transcript is the source of truth. This is deliberately stricter than generic
LLM self-critique: unsupported owners/deadlines/decisions are not allowed through.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from refiner import LLMError, chat_json, get_model, make_llm_client

ProgressCB = Optional[Callable[[float, str], None]]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "this", "that", "it", "as", "at", "by", "from", "be", "will", "would", "could", "should", "we", "they",
    "he", "she", "i", "you", "our", "their", "his", "her", "them", "do", "does", "did", "has", "have", "had",
}


class VerificationFinding(BaseModel):
    claim_id: str
    claim_type: str
    claim: str
    verdict: str = "insufficient"  # supported | contradicted | insufficient
    evidence_quote: str = ""
    evidence_start_s: Optional[float] = None
    evidence_end_s: Optional[float] = None
    reason: str = ""
    verified: bool = False


@dataclass
class _Turn:
    speaker: Optional[str]
    text: str
    start: float
    end: float


SYSTEM_VERIFY = """\
You are an independent evidence verifier for a meeting record.
You are NOT given the draft meeting record. You receive only ONE claim and transcript evidence passages.
The transcript is the sole source of truth.

Classify the claim:
- supported: the evidence directly supports the claim (reasonable paraphrase is allowed)
- contradicted: the evidence directly says something incompatible with the claim
- insufficient: the evidence does not establish the claim

Rules:
1. Never use outside knowledge.
2. Never fill missing owners, deadlines, names, numbers or decisions.
3. A suggestion is not an agreement. A discussion is not a decision.
4. If a person is not explicitly assigned/committed, do not treat them as the owner.
5. If a deadline is not explicitly stated for that task, it is unsupported.
6. For a claim containing multiple facts, mark it supported only if ALL material facts are supported.
7. Copy a short evidence quote EXACTLY from the supplied transcript when supported or contradicted.
8. If the evidence is ambiguous, choose insufficient rather than guessing.

Return ONLY JSON:
{"verdict":"supported|contradicted|insufficient","evidence_quote":"exact quote or empty","reason":"brief reason"}
"""

SYSTEM_REVISE = """\
You are the final conservative editor of a meeting record.
The transcript is the source of truth. A draft record and independent verification findings are provided.
Revise ONLY where verification shows that a draft claim is unsupported or contradicted.

Rules:
- Never invent replacement facts.
- Preserve supported wording and useful detail.
- Remove unsupported claims rather than guessing.
- For an unsupported/contradicted decision, remove that decision unless the transcript evidence supports a weaker
  proposal; if so, retain it only as a proposal.
- For an unsupported/contradicted action item, remove it unless the transcript evidence supports the task but not its
  owner/deadline; then keep the task and set missing fields to "Unspecified".
- Summary/minutes must contain only claims that are supported by evidence.
- Do not add facts from your own knowledge.
- Return the same JSON schema as the supplied MeetingRecord.
"""


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def _turns_from(transcript: Any) -> list[_Turn]:
    if isinstance(transcript, str):
        blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", transcript) if b.strip()]
        out = []
        for b in blocks:
            m = re.match(r"^\s*(Speaker\s+\d+)\s*:\s*(.*)$", b, re.I)
            out.append(_Turn(m.group(1) if m else None, m.group(2) if m else b, 0.0, 0.0))
        return out
    items = getattr(transcript, "turns", transcript)
    out = []
    for t in items:
        out.append(_Turn(getattr(t, "speaker", None), str(getattr(t, "text", "")),
                         float(getattr(t, "start", 0.0) or 0.0), float(getattr(t, "end", 0.0) or 0.0)))
    return [t for t in out if t.text.strip()]


def _retrieve(turns: list[_Turn], claim: str, k: int = 3) -> list[_Turn]:
    cw = _words(claim)
    scored: list[tuple[float, int]] = []
    for i, t in enumerate(turns):
        tw = _words(t.text)
        overlap = len(cw & tw)
        # Reward exact phrases / names and technical tokens. Keep a small base for short claims.
        phrase = 0.0
        norm_c = " ".join(_WORD.findall(claim.lower()))
        norm_t = " ".join(_WORD.findall(t.text.lower()))
        if norm_c and norm_c in norm_t:
            phrase = 8.0
        score = overlap + phrase
        if score:
            scored.append((score, i))
    scored.sort(reverse=True)
    chosen: list[int] = []
    for _, i in scored:
        # include adjacent turn for conversational context
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(turns) and j not in chosen:
                chosen.append(j)
        if len(chosen) >= k * 3:
            break
    return [turns[i] for i in chosen[:k * 3]]


def _claim_list(rec: Any, max_claims: int = 24) -> list[tuple[str, str, str]]:
    """Create atomic claims so missing owners/deadlines are verified independently.

    This is important: ``Owner: Unspecified`` is not itself a transcript fact and must not
    cause an otherwise valid task to fail verification.
    """
    claims: list[tuple[str, str, str]] = []
    for i, s in enumerate(re.split(r"(?<=[.!?])\s+", rec.summary or "")):
        s = s.strip()
        if len(s) >= 15:
            claims.append((f"summary.{i}", "summary", s))
    for si, sec in enumerate(rec.minutes):
        for pi, p in enumerate(sec.points):
            p = str(p).strip()
            if len(p) >= 12:
                claims.append((f"minutes.{si}.{pi}", "minutes", p))

    for i, d in enumerate(rec.decisions):
        claims.append((f"decision.{i}", "decision", d.statement))
        if d.status == "agreed":
            claims.append((f"decision.{i}.agreement", "decision_agreement",
                           f"The group explicitly agreed/settled this decision: {d.statement}"))

    for i, a in enumerate(rec.action_items):
        claims.append((f"action.{i}", "action_item", f"The meeting assigned or proposed this task: {a.description}"))
        if a.owner != "Unspecified":
            claims.append((f"action.{i}.owner", "action_owner", f"{a.owner} is explicitly responsible for: {a.description}"))
        if a.deadline != "Unspecified":
            claims.append((f"action.{i}.deadline", "action_deadline", f"The deadline for '{a.description}' is: {a.deadline}"))

    return claims[:max_claims]


def _verify_one(client, model: str, item: tuple[str, str, str], evidence: list[_Turn]) -> VerificationFinding:
    cid, ctype, claim = item
    evidence_text = "\n\n".join(
        f"[{t.start:.1f}-{t.end:.1f}] {t.speaker + ': ' if t.speaker else ''}{t.text}" for t in evidence
    ) or "[NO MATCHING TRANSCRIPT EVIDENCE FOUND]"
    prompt = f"CLAIM:\n{claim}\n\nTRANSCRIPT EVIDENCE:\n{evidence_text}"
    try:
        data = chat_json(client, model, SYSTEM_VERIFY, prompt, temperature=0.0, max_tokens=500)
        verdict = str(data.get("verdict", "insufficient")).strip().lower()
        if verdict not in {"supported", "contradicted", "insufficient"}:
            verdict = "insufficient"
        quote = str(data.get("evidence_quote", "") or "").strip()
        # only accept a quote that literally appears in the supplied evidence
        joined = " ".join(t.text for t in evidence)
        if quote and quote not in joined:
            quote = ""
            verdict = "insufficient"
        # Find the evidence timestamp by substring or best overlap.
        start = end = None
        if quote:
            for t in evidence:
                if quote in t.text:
                    start, end = t.start, t.end
                    break
        return VerificationFinding(claim_id=cid, claim_type=ctype, claim=claim, verdict=verdict,
                                    evidence_quote=quote, evidence_start_s=start, evidence_end_s=end,
                                    reason=str(data.get("reason", "")).strip(), verified=verdict == "supported")
    except Exception as e:
        return VerificationFinding(claim_id=cid, claim_type=ctype, claim=claim, verdict="insufficient",
                                    reason=f"Verifier failed: {e}", verified=False)


def _revise_record(client, model: str, rec: Any, findings: list[VerificationFinding]) -> Any:
    # Import lazily to avoid a circular import at module load.
    from extractor import MeetingRecord
    draft = rec.model_dump(exclude={"warnings", "model", "meeting_duration_minutes", "speaker_labels_used"})
    compact_findings = [f.model_dump(exclude={"verified"}) for f in findings]
    prompt = "DRAFT RECORD:\n" + __import__("json").dumps(draft, ensure_ascii=False) + \
             "\n\nVERIFICATION FINDINGS:\n" + __import__("json").dumps(compact_findings, ensure_ascii=False) + \
             "\n\nReturn the revised MeetingRecord JSON now."
    schema = MeetingRecord.model_json_schema()
    data = chat_json(client, model, SYSTEM_REVISE, prompt, temperature=0.0, max_tokens=6000,
                     schema=schema, schema_name="verified_meeting_record")
    for key in ("warnings", "model", "meeting_duration_minutes", "speaker_labels_used"):
        data.pop(key, None)
    return MeetingRecord.model_validate(data)


def verify_and_revise(rec: Any, transcript: Any, api_key: Optional[str] = None,
                      base_url: Optional[str] = None, model: Optional[str] = None,
                      progress_cb: ProgressCB = None, max_claims: int = 24) -> tuple[Any, list[VerificationFinding]]:
    report = progress_cb or (lambda f, m: None)
    turns = _turns_from(transcript)
    claims = _claim_list(rec, max_claims=max_claims)
    if not claims:
        return rec, []
    client = make_llm_client(api_key, base_url)
    model_id = get_model(model)

    report(0.05, f"Planning {len(claims)} transcript-grounded verification checks...")
    jobs = [(item, _retrieve(turns, item[2])) for item in claims]
    findings: list[VerificationFinding] = []
    # Factored CoVe: independent verification calls. Parallelism keeps wall-clock latency reasonable.
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(jobs)))) as ex:
        futs = [ex.submit(_verify_one, client, model_id, item, evidence) for item, evidence in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            findings.append(fut.result())
            report(0.10 + 0.55 * n / len(futs), f"Verified claim {n}/{len(futs)}...")
    findings.sort(key=lambda f: claims.index(next(c for c in claims if c[0] == f.claim_id)))

    report(0.68, "Cross-checking verification findings and revising the draft record...")
    try:
        revised = _revise_record(client, model_id, rec, findings)
    except LLMError:
        # Safe fallback: retain the deterministic grounded record rather than failing the whole pipeline.
        revised = rec
        revised.warnings.append("CoVe revision call failed; deterministic grounding/confidence results were retained.")
    except Exception as e:
        revised = rec
        revised.warnings.append(f"CoVe revision failed ({type(e).__name__}); deterministic grounding/confidence results were retained.")

    # Re-run the existing deterministic grounding after revision. This is important because the revision LLM is not trusted.
    from extractor import ground_record, turns_to_input
    revised = ground_record(revised, turns_to_input([(t.speaker, t.text) for t in turns]))

    supported = sum(f.verdict == "supported" for f in findings)
    contradicted = sum(f.verdict == "contradicted" for f in findings)
    insufficient = sum(f.verdict == "insufficient" for f in findings)
    revised.warnings.append(
        f"Transcript-grounded CoVe checked {len(findings)} claims: {supported} supported, "
        f"{contradicted} contradicted, {insufficient} insufficient."
    )
    report(1.0, "Hallucination verification complete.")
    return revised, findings
