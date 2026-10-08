# Domain knowledge base

Each `*.json` file is one domain:

```json
{"domain": "name", "title": "...", "description": "...",
 "cues": ["words that signal this domain in a transcript"],
 "terms": [{"term": "Kubernetes", "expansion": "optional meaning",
            "aliases": ["cooper netties"], "note": "optional"}]}
```

* A domain is switched on when at least 3 of its `cues` appear in the transcript (cues match word prefixes).
* `aliases` are known mis-hearings; they are only used when the domain is on.
* Put your own terms in `custom/*.txt` (see `custom/my_terms.txt`) - those are always on.
* Edits are picked up on the next run; no restart needed.
