"""
domain_kb.py  --  Domain knowledge retrieval for the refiner (retrieval-augmented proofreading)

Why this exists
---------------
An LLM that sees a transcript with no context cannot know that "Army of Springfield" is really
"RM of Springfield" or that "cooper netties" is "Kubernetes". This module gives the refiner that
context *before* it edits anything, without a vector database:

   transcript text
        |
        |-- 1. keyphrases ........ KeyBERT (semantic) if installed, otherwise a frequency fallback
        |-- 2. domain routing .... which dictionaries apply? (cue words found in the transcript)
        |-- 3. retrieval ......... every 1-4 word span of the transcript is compared with every dictionary
        |                          term and its known mis-hearings, by EXACT alias match, PHONETIC match
        |                          (Metaphone - speech errors are errors of *sound*) and character similarity
        |-- 4. self-consistency .. the same word spelled two ways inside this transcript
        |                          ("Tarian" / "Terrian", "Recyclers form" / "Recyclers Forum")
        '-- 5. glossary ........... a short, relevant glossary + suspect spans, injected into the LLM prompt

Dictionaries live in ./domain_kb/*.json (see README there). Add your own terms in ./domain_kb/custom/*.txt
or through the "domain hints" box in the app; those are always active.

Everything here is deterministic and LLM-free. refiner.py optionally adds an LLM "briefing" on top
(DomainContext.add_briefing) and uses DomainContext.supports() to decide which edits are well-founded.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

KB_DIR = Path(os.getenv("DOMAIN_KB_DIR") or Path(__file__).resolve().with_name("domain_kb"))
MIN_CUE_HITS = 3          # a domain is switched on when this many of its cue words occur
MAX_ACTIVE_DOMAINS = 4
MAX_GLOSSARY = 60
MAX_HINTS = 80

# Optional accelerators -------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz, process as _rf_process
except ImportError:                                   # pragma: no cover
    _rf_fuzz = _rf_process = None
try:
    import jellyfish as _jf
except ImportError:                                   # pragma: no cover
    _jf = None

# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------
_APOS = {ord("’"): "'", ord("‘"): "'"}
_FLAT_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’][A-Za-z0-9À-ɏ]+)*")

# Words that carry grammar, not content. Negations, modals and numbers are deliberately NOT here:
# swapping those changes meaning, and refiner.py protects them separately.
GRAMMAR_WORDS = frozenset("""
a an the and or but if so of to in on at for from by with as is are was were be been being am do does did
have has had i you he she it we they me him her us them my your his its our their this that these those
there here than then too very just also into over under about up down out off who whom whose which what
when where why how all any some more most other such own same
""".split())
_STOP = GRAMMAR_WORDS | frozenset("okay ok well yeah yes please thank thanks".split())


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.translate(_APOS).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def flat(s: str) -> str:
    """lower-case, accent-free, punctuation-free, single-spaced: the canonical comparison form"""
    return " ".join(_FLAT_RE.findall(fold(s)))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _skeleton(s: str) -> str:
    """tiny consonant-skeleton phonetic key, used only when jellyfish is not installed"""
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("kn", "n"), ("wr", "r"), ("wh", "w"), ("x", "ks"),
                 ("q", "k"), ("z", "s"), ("c", "k")):
        s = s.replace(a, b)
    s = s[:1] + re.sub(r"[aeiouyhw]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s).upper()


@lru_cache(maxsize=100_000)
def phonetic(s: str) -> str:
    letters = re.sub(r"[^a-z]", "", fold(s))
    if not letters:
        return ""
    if _jf is not None:
        try:
            return _jf.metaphone(letters)
        except Exception:                             # pragma: no cover
            pass
    return _skeleton(letters)


def char_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _rf_fuzz is not None:
        return _rf_fuzz.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
@dataclass
class Term:
    canonical: str
    domain: str
    expansion: str = ""
    aliases: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def key(self) -> str:
        return flat(self.canonical)

    def describe(self) -> str:
        bits = [b for b in (self.expansion, self.note) if b]
        return "; ".join(bits)


@dataclass
class Domain:
    name: str
    title: str
    description: str
    cues: list[str]
    terms: list[Term]
    always_on: bool = False


@dataclass
class Hint:
    span: str                  # the words as they appear in the transcript
    suggestion: str            # what they may really be
    score: float
    source: str                # dictionary-alias | dictionary-fuzzy | user-alias | user-fuzzy | briefing
    domain: str = ""
    note: str = ""
    count: int = 1
    term: Optional[Term] = field(default=None, repr=False)

    def key(self) -> str:
        return flat(self.span)


@dataclass
class GlossaryEntry:
    term: str
    expansion: str = ""
    note: str = ""
    domain: str = ""


@dataclass
class VariantCluster:
    kind: str                                  # "name" | "phrase"
    forms: list[tuple[str, int]]               # (spelling, occurrences)

    def text(self) -> str:
        return " / ".join(f"{f} ({n}x)" for f, n in self.forms)


@dataclass
class DomainContext:
    active_domains: list[dict] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    keyphrase_method: str = ""
    glossary: list[GlossaryEntry] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    variants: list[VariantCluster] = field(default_factory=list)
    user_notes: str = ""
    briefing: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    trusted_pairs: set = field(default_factory=set, repr=False)
    trusted_tokens: set = field(default_factory=set, repr=False)
    variant_counts: dict = field(default_factory=dict, repr=False)

    # ---- evidence used by the refiner's edit validator --------------------------------
    def supports(self, old_flat: str, new_flat: str) -> bool:
        """True when replacing `old_flat` by `new_flat` is backed by evidence other than the LLM's say-so."""
        if (old_flat, new_flat) in self.trusted_pairs:
            return True
        new_tokens = new_flat.split()
        if new_tokens and any(t in self.trusted_tokens for t in new_tokens) and \
                all(t in self.trusted_tokens or t in GRAMMAR_WORDS for t in new_tokens):
            # the replacement is made of glossary words, but it must still SOUND like what was heard
            # (otherwise 'Council' -> 'Councillor' or 'budget' -> 'RM' would slip through)
            o, n = old_flat.replace(" ", ""), new_flat.replace(" ", "")
            if o and n and not (n.startswith(o) or o.startswith(n)):
                po, pn = phonetic(old_flat), phonetic(new_flat)
                if char_ratio(o, n) >= 0.6 or (po and pn and char_ratio(po, pn) >= 0.8):
                    return True
        counts = self.variant_counts.get(old_flat)
        if counts and new_flat in counts and counts[new_flat] > counts[old_flat]:
            return True                      # same word spelled both ways; this spelling clearly dominates
        return False

    # ---- briefing (LLM-assisted, optional) ---------------------------------------------
    def add_briefing(self, briefing: Optional[dict], text: str) -> None:
        if not isinstance(briefing, dict):
            return
        ftext = " " + flat(text) + " "

        def _strs(key: str, n: int = 25) -> list[str]:
            v = briefing.get(key) or []
            return [str(x).strip() for x in v if str(x).strip()][:n] if isinstance(v, list) else []

        clean: dict[str, Any] = {
            "domain": str(briefing.get("domain") or "").strip()[:160],
            "meeting_type": str(briefing.get("meeting_type") or "").strip()[:160],
            "organizations": _strs("organizations"), "people": _strs("people"),
            "places": _strs("places"), "key_terms": _strs("key_terms"),
            "suspected_misrecognitions": [],
        }
        for item in (briefing.get("suspected_misrecognitions") or [])[:40]:
            if not isinstance(item, dict):
                continue
            heard, likely = str(item.get("heard") or "").strip(), str(item.get("likely") or "").strip()
            fh, fl = flat(heard), flat(likely)
            if not fh or not fl or fh == fl or f" {fh} " not in ftext:
                continue                      # the "heard" words must really occur in the transcript
            why = str(item.get("why") or "").strip()[:160]
            self.hints.append(Hint(heard, likely, 0.9, "briefing", "briefing", why))
            self.trusted_pairs.add((fh, fl))
            self.trusted_tokens.update(fl.split())
            clean["suspected_misrecognitions"].append({"heard": heard, "likely": likely, "why": why})
        self.briefing = clean

    # ---- prompt material ---------------------------------------------------------------
    def glossary_block(self, max_chars: int = 3800) -> str:
        b = self.briefing or {}
        lines: list[str] = []
        ctx_bits = []
        if b.get("domain"):
            ctx_bits.append(f"domain: {b['domain']}")
        elif self.active_domains:
            ctx_bits.append("domain: " + ", ".join(d["title"] for d in self.active_domains))
        if b.get("meeting_type"):
            ctx_bits.append(f"type: {b['meeting_type']}")
        for label, key in (("organisations", "organizations"), ("places", "places"), ("people", "people")):
            if b.get(key):
                ctx_bits.append(f"{label} mentioned: " + ", ".join(b[key][:15]))
        if ctx_bits:
            lines += ["MEETING CONTEXT (retrieved automatically - use it to decide what was really said):"]
            lines += [f"- {x}" for x in ctx_bits]
        if self.user_notes:
            lines += ["", "NOTES FROM THE USER: " + self.user_notes]
        if self.glossary:
            lines += ["", "GLOSSARY (authoritative spellings - use these exact forms):"]
            for g in self.glossary:
                extra = "; ".join(x for x in (g.expansion, g.note) if x)
                lines.append(f"- {g.term}" + (f" - {extra}" if extra else ""))
        if self.variants:
            lines += ["", "THE SAME WORD IS SPELLED DIFFERENTLY IN THIS TRANSCRIPT (variants of one word):"]
            lines += [f"- {v.text()}" for v in self.variants[:12]]
            lines += ["  Unify them only when one spelling is clearly right (glossary, or it appears far more often)."]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars].rsplit("\n", 1)[0] + "\n- ..."

    def hints_for(self, text: str, limit: int = 12) -> list[Hint]:
        ft = " " + flat(text) + " "
        sel = [h for h in self.hints if f" {h.key()} " in ft]
        sel.sort(key=lambda h: (-h.score, -h.count, h.span))
        out, seen = [], set()
        for h in sel:
            if (h.key(), flat(h.suggestion)) not in seen:
                seen.add((h.key(), flat(h.suggestion)))
                out.append(h)
        return out[:limit]

    def variants_for(self, text: str) -> list[VariantCluster]:
        ft = " " + flat(text) + " "
        return [v for v in self.variants if any(f" {flat(f)} " in ft for f, _ in v.forms)]

    def batch_block(self, text: str) -> str:
        hs, vs = self.hints_for(text), self.variants_for(text)
        lines: list[str] = []
        if hs:
            lines.append("SUSPECT SPANS IN THESE PASSAGES (heard -> likely). Change one only if the sentence "
                         "really makes sense with the substitution:")
            for h in hs:
                extra = f"  [{h.note}]" if h.note else ""
                lines.append(f'- "{h.span}" -> "{h.suggestion}"{extra}')
        if vs:
            lines.append("SPELLING VARIANTS OF ONE WORD IN THESE PASSAGES: " + "; ".join(v.text() for v in vs[:6]))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "active_domains": self.active_domains,
            "keyphrases": self.keyphrases,
            "keyphrase_method": self.keyphrase_method,
            "glossary": [g.__dict__ for g in self.glossary],
            "suspect_spans": [{"heard": h.span, "likely": h.suggestion, "score": h.score, "source": h.source,
                               "domain": h.domain, "note": h.note, "occurrences": h.count} for h in self.hints],
            "spelling_variants": [{"kind": v.kind, "forms": [{"spelling": f, "count": n} for f, n in v.forms]}
                                  for v in self.variants],
            "user_notes": self.user_notes,
            "briefing": self.briefing,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Dictionary loading
# --------------------------------------------------------------------------------------
def parse_term_spec(item: str, domain: str) -> Optional[Term]:
    """'Term' | 'alias -> Term' | 'Term | alias one; alias two | note'"""
    item = item.strip()
    if not item or item.startswith("#"):
        return None
    if "->" in item:
        alias, _, term = item.partition("->")
        alias, term = alias.strip(), term.strip()
        return Term(term, domain, aliases=[alias]) if alias and term else None
    if "|" in item:
        parts = [p.strip() for p in item.split("|")]
        term = parts[0]
        aliases = [a.strip() for a in parts[1].split(";")] if len(parts) > 1 else []
        note = parts[2] if len(parts) > 2 else ""
        return Term(term, domain, aliases=[a for a in aliases if a], note=note) if term else None
    return Term(item.strip(" .;,"), domain)


_LABEL_SPLIT = re.compile(
    r"[,;\n]|\.\s+|\b(?:terms?|people|names?|places?|orgs?|organi[sz]ations?|acronyms?|glossary)\s*:", re.I)


def parse_user_hints(hints: str) -> tuple[list[Term], str]:
    """-> (terms, notes). Free text is kept verbatim as notes; short comma/line items become terms."""
    hints = (hints or "").strip()
    if not hints:
        return [], ""
    terms: list[Term] = []
    for line in hints.splitlines():
        if "->" in line or "|" in line:
            t = parse_term_spec(line, "user")
            if t:
                terms.append(t)
            continue
        for chunk in _LABEL_SPLIT.split(line):
            chunk = chunk.strip(" .:;-")
            if 2 < len(chunk) <= 60 and len(chunk.split()) <= 6:
                terms.append(Term(chunk, "user"))
    return terms, hints[:800]


def _kb_signature() -> tuple:
    files: list[Path] = []
    try:
        files = sorted(KB_DIR.glob("*.json")) + sorted((KB_DIR / "custom").glob("*"))
    except OSError:
        pass
    sig = []
    for p in files:
        try:
            if p.is_file():
                sig.append((str(p), p.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


@lru_cache(maxsize=4)
def _load_domains(signature: tuple) -> tuple[Domain, ...]:
    domains: list[Domain] = []
    for path_str, _ in signature:
        p = Path(path_str)
        try:
            if p.suffix.lower() == ".json":
                doc = json.loads(p.read_text(encoding="utf-8"))
                name = str(doc.get("domain") or p.stem)
                terms = []
                for t in doc.get("terms", []):
                    if isinstance(t, dict) and t.get("term"):
                        terms.append(Term(str(t["term"]), name, str(t.get("expansion", "")),
                                          [str(a) for a in t.get("aliases", [])], str(t.get("note", ""))))
                always = p.parent.name == "custom"
                domains.append(Domain(name, str(doc.get("title") or name), str(doc.get("description", "")),
                                      [flat(str(c)) for c in doc.get("cues", []) if flat(str(c))], terms, always))
            elif p.suffix.lower() in (".txt", ".csv") and p.parent.name == "custom":
                name = f"custom:{p.stem}"
                terms = [t for t in (parse_term_spec(l, name) for l in p.read_text(encoding="utf-8").splitlines()) if t]
                if terms:
                    domains.append(Domain(name, f"Custom terms ({p.stem})", "", [], terms, True))
        except Exception as e:                                # a broken file must not break the pipeline
            logger.warning("Could not load domain file %s: %s", p, e)
    return tuple(domains)


def load_domains() -> tuple[Domain, ...]:
    return _load_domains(_kb_signature())


# --------------------------------------------------------------------------------------
# Keyphrases (KeyBERT if available)
# --------------------------------------------------------------------------------------
_KEYBERT: dict[str, Any] = {"tried": False, "model": None, "error": None}


def _get_keybert():
    if _KEYBERT["tried"]:
        return _KEYBERT["model"]
    _KEYBERT["tried"] = True
    try:
        from keybert import KeyBERT                      # needs: pip install keybert sentence-transformers
        _KEYBERT["model"] = KeyBERT(model=os.getenv("KEYBERT_MODEL", "all-MiniLM-L6-v2"))
    except Exception as e:
        _KEYBERT["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return _KEYBERT["model"]


def _frequency_keyphrases(text: str, top_n: int) -> list[str]:
    toks = tokenize(text)
    counts: Counter = Counter()
    caps: Counter = Counter()
    for i, t in enumerate(toks):
        ft = fold(t)
        if ft in _STOP or len(ft) < 3 or ft.isdigit():
            continue
        counts[ft] += 1
        if t[0].isupper() and i > 0:
            caps[ft] += 1
    for a, b in zip(toks, toks[1:]):
        fa, fb = fold(a), fold(b)
        if fa in _STOP or fb in _STOP or len(fa) < 3 or len(fb) < 3:
            continue
        counts[f"{fa} {fb}"] += 1
    scored = []
    for k, c in counts.items():
        if c < 2 and " " in k:
            continue
        cap_ratio = caps.get(k, 0) / c if " " not in k else 0.0
        scored.append((c * (1 + cap_ratio) * math.log(2 + len(k)), k))
    scored.sort(reverse=True)
    out: list[str] = []
    for _, k in scored:
        if not any(k in o or o in k for o in out):
            out.append(k)
        if len(out) >= top_n:
            break
    return out


def extract_keyphrases(text: str, top_n: int = 25, use_keybert: bool = True) -> tuple[list[str], str, Optional[str]]:
    """-> (phrases, method, problem). method is 'KeyBERT' or 'frequency'."""
    problem = None
    if use_keybert:
        kb = _get_keybert()
        if kb is not None:
            try:
                pairs = kb.extract_keywords(text[:60000], keyphrase_ngram_range=(1, 3), stop_words="english",
                                            use_mmr=True, diversity=0.5, top_n=top_n)
                return [p for p, _ in pairs], "KeyBERT", None
            except Exception as e:
                problem = f"KeyBERT failed ({type(e).__name__}: {str(e)[:100]})"
        else:
            problem = f"KeyBERT unavailable ({_KEYBERT['error'] or 'not installed'})"
    return _frequency_keyphrases(text, top_n), "frequency", problem


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------
def _collect_ngrams(tokens: list[str], max_n: int = 4) -> dict[str, tuple[str, int]]:
    ft = [flat(t) for t in tokens]
    out: dict[str, list] = {}
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            parts = ft[i:i + n]
            if any(not p or " " in p for p in parts):
                continue
            key = " ".join(parts)
            if key in out:
                out[key][1] += 1
            else:
                out[key] = [" ".join(tokens[i:i + n]), 1]
    return {k: (v[0], v[1]) for k, v in out.items()}


class _Index:
    def __init__(self, terms: list[Term]):
        self.alias_exact: dict[str, list[Term]] = defaultdict(list)
        self.canon_keys: dict[str, Term] = {}
        self.forms: list[tuple[str, str, Term, int]] = []     # (no-space text, phonetic, term, n words)
        for t in terms:
            self.canon_keys[t.key] = t
            for form, is_alias in [(t.canonical, False)] + [(a, True) for a in t.aliases]:
                fk = flat(form)
                if not fk:
                    continue
                if is_alias:
                    self.alias_exact[fk].append(t)
                self.forms.append((fk.replace(" ", ""), phonetic(fk), t, fk.count(" ") + 1))
        self.form_strs = [f[0] for f in self.forms]
        self.form_phons = [f[1] for f in self.forms]
        self.phon_map: dict[str, list[int]] = defaultdict(list)
        for i, ph in enumerate(self.form_phons):
            if ph:
                self.phon_map[ph].append(i)


def _put(store: dict, h: Hint) -> None:
    k = (h.key(), flat(h.suggestion))
    old = store.get(k)
    if old is None:
        store[k] = h
    else:
        old.score = max(old.score, h.score)
        old.count = max(old.count, h.count)


def _retrieve(ngrams: dict[str, tuple[str, int]], idx: _Index, source: str, min_score: float) -> list[Hint]:
    out: dict[tuple[str, str], Hint] = {}
    for g, (span, cnt) in ngrams.items():
        parts = g.split(" ")
        for t in idx.alias_exact.get(g, ()):
            if g != t.key:
                _put(out, Hint(span, t.canonical, 1.0, f"{source}-alias", t.domain, t.describe(), cnt, t))
        if parts[0] in _STOP or parts[-1] in _STOP or g in idx.canon_keys:
            continue
        gs = g.replace(" ", "")
        if len(gs) < 4 or any(ch.isdigit() for ch in gs):
            continue
        pg = phonetic(g)
        cand: set[int] = set()
        if _rf_process is not None:
            cand.update(i for _, _, i in _rf_process.extract(gs, idx.form_strs, scorer=_rf_fuzz.ratio,
                                                              score_cutoff=80, limit=5))
            if len(pg) >= 3:
                cand.update(i for _, _, i in _rf_process.extract(pg, idx.form_phons, scorer=_rf_fuzz.ratio,
                                                                  score_cutoff=80, limit=8))
        else:
            cand.update(idx.phon_map.get(pg, ()))
        for i in cand:
            ns, pf, t, nt = idx.forms[i]
            if not (len(parts) == nt or len(parts) == nt + 1) or gs == ns or gs.rstrip("s") == ns.rstrip("s"):
                continue
            if ns.startswith(gs) or gs.startswith(ns):      # 'Council' vs 'Councillor': a real word, not an error
                continue
            c = char_ratio(gs, ns)
            p = char_ratio(pg, pf) if pg and pf else 0.0
            ok = (p >= 0.82 and c >= 0.60 and pg[:1] == pf[:1]) or (c >= 0.88 and gs[0] == ns[0])
            score = round(max(c, p * 0.95), 3)
            if ok and score >= min_score:
                _put(out, Hint(span, t.canonical, score, f"{source}-fuzzy", t.domain, t.describe(), cnt, t))
    return sorted(out.values(), key=lambda h: (-h.score, h.span))


# --------------------------------------------------------------------------------------
# Self-consistency: the same word spelled two ways inside one transcript
# --------------------------------------------------------------------------------------
def find_variants(text: str, max_clusters: int = 25) -> tuple[list[VariantCluster], dict]:
    # --- 1) capitalised words (names / terms) that are near-duplicates of each other
    counts: Counter = Counter()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if len(tok) < 4 or not tok[0].isupper() or not tok.isalpha():
            continue
        before = text[:m.start()].rstrip()
        if not before or before[-1] in ".?!":
            continue                                     # sentence start: capital letter proves nothing
        counts[tok] += 1
    forms = list(counts)
    parent = {f: f for f in forms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    folded = {f: fold(f) for f in forms}
    phon = {f: phonetic(f) for f in forms}
    for i, a in enumerate(forms):
        for b in forms[i + 1:]:
            fa, fb = folded[a], folded[b]
            if fa == fb:
                parent[find(a)] = find(b)
                continue
            c = char_ratio(fa, fb)
            p = char_ratio(phon[a], phon[b]) if phon[a] and phon[b] else 0.0
            if (p >= 0.80 and c >= 0.60 and phon[a][:1] == phon[b][:1]) or (c >= 0.85 and fa[0] == fb[0]):
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for f in forms:
        groups[find(f)].append(f)
    clusters: list[VariantCluster] = []
    variant_counts: dict[str, dict[str, int]] = {}
    for members in groups.values():
        merged: Counter = Counter()
        for f in members:
            merged[f] += counts[f]
        spellings = {}
        for f, n in merged.items():                      # merge pure-case duplicates
            spellings[folded[f]] = spellings.get(folded[f], 0) + n
        if len(spellings) < 2:
            continue
        clusters.append(VariantCluster("name", sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))))
        cmap = {flat(k): v for k, v in spellings.items()}
        for k in cmap:
            variant_counts[k] = cmap

    # --- 2) same preceding content word, followers that are spelled almost the same ("recyclers form/forum")
    toks = [fold(t) for t in tokenize(text)]
    followers: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(toks, toks[1:]):
        if len(a) >= 4 and a not in _STOP and len(b) >= 4 and b not in _STOP and a.isalpha() and b.isalpha():
            followers[a][b] += 1
    for left, fc in followers.items():
        words = list(fc)
        for i, x in enumerate(words):
            for y in words[i + 1:]:
                if x[0] == y[0] and char_ratio(x, y) >= 0.75:
                    clusters.append(VariantCluster("phrase", sorted(
                        [(f"{left} {x}", fc[x]), (f"{left} {y}", fc[y])], key=lambda kv: (-kv[1], kv[0]))))
    clusters.sort(key=lambda v: -sum(n for _, n in v.forms))
    return clusters[:max_clusters], variant_counts


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def _cue_hit(cue: str, ftext: str) -> bool:
    return (f" {cue} " in ftext) if " " in cue else (f" {cue}" in ftext)     # single words match as prefixes


def build_context(text: str, user_hints: str = "", use_dictionaries: bool = True,
                  use_keybert: bool = True) -> DomainContext:
    ctx = DomainContext()
    tokens = tokenize(text)
    ftext = " " + flat(text) + " "
    ngrams = _collect_ngrams(tokens)

    user_terms, ctx.user_notes = parse_user_hints(user_hints)

    terms: list[Term] = []
    if use_dictionaries:
        ctx.keyphrases, ctx.keyphrase_method, problem = extract_keyphrases(text, use_keybert=use_keybert)
        if problem:
            ctx.notes.append(problem + "; used the frequency-based keyphrase fallback instead.")
        scored = []
        for d in load_domains():
            hits = [c for c in d.cues if _cue_hit(c, ftext)]
            if d.always_on or len(hits) >= MIN_CUE_HITS:
                scored.append((d, hits))
        scored.sort(key=lambda dh: (-int(dh[0].always_on), -len(dh[1])))
        always = [x for x in scored if x[0].always_on]
        regular = [x for x in scored if not x[0].always_on][:MAX_ACTIVE_DOMAINS]
        for d, hits in always + regular:
            terms.extend(d.terms)
            ctx.active_domains.append({"name": d.name, "title": d.title, "cue_hits": hits[:8]})
        if not ctx.active_domains:
            ctx.notes.append("No built-in domain matched this transcript; only user hints, the LLM briefing and "
                             "in-transcript consistency checks are used.")

    hints: list[Hint] = []
    if terms:
        hints += _retrieve(ngrams, _Index(terms), "dictionary", 0.80)
    if user_terms:
        hints += _retrieve(ngrams, _Index(user_terms), "user", 0.75)
    hints.sort(key=lambda h: (-h.score, -h.count))
    ctx.hints = hints[:MAX_HINTS]

    # glossary: user terms, terms with a suspect span, then dictionary terms that already occur correctly
    seen: set[str] = set()

    def add(term: Term) -> None:
        if term.key and term.key not in seen and len(ctx.glossary) < MAX_GLOSSARY:
            seen.add(term.key)
            ctx.glossary.append(GlossaryEntry(term.canonical, term.expansion, term.note, term.domain))
            ctx.trusted_tokens.update(term.key.split())

    for t in user_terms:
        add(t)
    for h in ctx.hints:
        if h.term is not None:
            add(h.term)
    for t in terms:
        if f" {t.key} " in ftext:
            add(t)
    for h in ctx.hints:
        if h.score >= 0.88:
            ctx.trusted_pairs.add((h.key(), flat(h.suggestion)))

    ctx.variants, ctx.variant_counts = find_variants(text)
    return ctx
