"""
RAG-chunk-text -> exact-node-name matching for the PBDS tool.

Two layers (see the design decision in project memory):

  Layer 1 - deterministic candidate generation (recall-first):
      BM25 over the node *descriptions* as the corpus, with the retrieved chunk
      text as the query. IDF naturally downweights generic terms shared by many
      parameters ("core", "power") and rewards rare, specific ones ("MOX",
      "enrichment"), which is exactly what short, overlapping labels need. A light
      stdlib singularizer aligns plurals ("pins" -> "pin"). Produces a shortlist.

  Layer 2 - LLM verification (precision):
      An LLM is asked which (if any) shortlisted parameter the passage actually
      discusses, with its answer constrained by a JSON schema to one of the
      shortlisted node names or null. This rejects mere keyword overlap and
      disambiguates near-duplicate descriptions.

The design is precision-leaning: it is better to miss a match than to expand the
report with an irrelevant parameter's sources/effects. A confirmed node name
feeds straight into PBDSManager.neighborhood for k-hop source/effect expansion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from qmix_report_writer.llm.format import Message
from .pbds_manager import ConnectedNode, PBDSManager

# Small English stopword set: enough to stop function words from polluting BM25
# without an NLP dependency. Domain terms are deliberately kept.
_STOPWORDS = frozenset(
    "a an the of for with in on to from at by as is are be of and or not no "
    "this that these those it its their our we into under over between through "
    "per via using used use within without than then so such each any all".split()
)

# Trailing "_F<row>" id stripped when falling back to a node name as its own text.
_F_SUFFIX_RE = re.compile(r"_F\d+$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# First {...} object in a response, used to recover JSON the model wrapped in
# markdown fences or surrounded with prose despite the response schema.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _loads_lenient(raw) -> Optional[dict]:
    """Parse a JSON object from a model response, tolerating fences/prose.

    The response schema usually yields clean JSON, but some models still wrap it
    in ```json ... ``` fences or add surrounding text. Returns None if no JSON
    object can be recovered.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        m = _JSON_OBJ_RE.search(s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None


def _singularize(token: str) -> str:
    """Very small, conservative singularizer (no NLP dependency).

    Handles the common plurals in the parameter sheet (pins->pin,
    assemblies->assembly) while leaving genuine words ending in 'ss' (mass,
    stress) and short tokens untouched.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, singularize."""
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        out.append(_singularize(tok))
    return out


def _humanize_node_name(name: str) -> str:
    """Fallback text for a node with no description: 'Pellet_diameter_F8' -> 'Pellet diameter'."""
    return _F_SUFFIX_RE.sub("", name).replace("_", " ").strip()


@dataclass
class Candidate:
    """A node shortlisted by the deterministic layer."""

    node: str          # exact node name
    description: str    # text the chunk was matched against
    score: float        # BM25 score of the chunk against this description


_VERIFIER_SYSTEM = (
    "You link a passage of technical text to nuclear-reactor design parameters. "
    "You are given a passage and a short list of candidate parameters (name + description). "
    "Return EVERY candidate the passage substantively discusses — i.e. is actually about "
    "that physical quantity, not merely sharing a word with its name. Return an empty list "
    "if none of the candidates is genuinely discussed. Prefer omitting a weak match over "
    "including it."
)


class NodeMatcher:
    """Match retrieved RAG chunks to exact PBDS node names.

    Builds a BM25 index over node descriptions once; reuse one instance across
    chunks. The LLM used for verification is injected per call so this stays
    decoupled from any particular model/agent wiring.
    """

    def __init__(
        self,
        manager: PBDSManager,
        top_k: int = 8,
        score_floor: float = 0.0,
        min_score_ratio: float = 0.25,
        max_neighbors: Optional[int] = 20,
    ):
        """
        Args:
            manager:         the PBDSManager whose graph/descriptions to match against.
            top_k:           maximum candidates kept in a shortlist.
            score_floor:     absolute minimum BM25 score for a candidate (exclusive).
                             Raise above 0 to skip the LLM entirely on chunks whose
                             best candidate is still weak (note: BM25 magnitude scales
                             with chunk length, so an absolute floor is workload-specific).
            min_score_ratio: relative floor — drop candidates scoring below this
                             fraction of the top candidate's score. Trims the tail of
                             a shortlist so the LLM only judges genuinely competitive
                             candidates. Set to 0 to disable.
            max_neighbors:   default cap on neighbours returned by expand(); None = no cap.
        """
        self.manager = manager
        self.top_k = top_k
        self.score_floor = score_floor
        self.min_score_ratio = min_score_ratio
        self.max_neighbors = max_neighbors

        descriptions = manager.descriptions()
        self._node_ids: List[str] = list(descriptions.keys())
        # Text each node is matched against: its description, or a humanized name
        # when the description is blank.
        self._texts: List[str] = [
            descriptions[n] or _humanize_node_name(n) for n in self._node_ids
        ]
        corpus_tokens = [_tokenize(t) for t in self._texts]
        # Guard empty tokenizations (e.g. a purely numeric label) so BM25 has a term.
        corpus_tokens = [toks if toks else [""] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(corpus_tokens)

    # ------------------------------------------------------------------ #
    # Layer 1 - deterministic shortlist
    # ------------------------------------------------------------------ #
    def shortlist(self, chunk_text: str, top_k: Optional[int] = None) -> List[Candidate]:
        """Return up to `top_k` candidate nodes for a chunk, best score first.

        Recall-first: this casts a wide net; the LLM verifier tightens it.
        Candidates are deduplicated by description — the workbook reuses the same
        label on several rows, and those nodes are indistinguishable to the verifier
        (and share neighbours), so keeping only the best-scoring one per description
        frees shortlist slots for genuinely different parameters.
        Returns an empty list when nothing scores above `score_floor`.
        """
        query = _tokenize(chunk_text)
        if not query:
            return []
        scores = self._bm25.get_scores(query)
        cap = self.top_k if top_k is None else top_k
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if not order:
            return []
        # Keep candidates above BOTH the absolute floor and a fraction of the top
        # score, so a clear winner trims its weaker neighbours from the shortlist.
        top_score = float(scores[order[0]])
        threshold = max(self.score_floor, top_score * self.min_score_ratio)
        out: List[Candidate] = []
        seen_descriptions: set = set()
        for i in order:
            if scores[i] <= threshold:
                break  # order is descending, so nothing better remains
            key = self._texts[i].strip().lower()
            if key in seen_descriptions:
                continue  # a higher-scoring node with this description is already in
            seen_descriptions.add(key)
            out.append(
                Candidate(
                    node=self._node_ids[i],
                    description=self._texts[i],
                    score=round(float(scores[i]), 4),
                )
            )
            if len(out) >= cap:
                break
        return out

    # ------------------------------------------------------------------ #
    # Layer 2 - LLM verification
    # ------------------------------------------------------------------ #
    @staticmethod
    def _verifier_schema(candidates: List[Candidate]) -> Dict:
        """JSON schema constraining the answer to a subset of the candidate node names."""
        return {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": [c.node for c in candidates]},
                },
                "reason": {"type": "string"},
            },
            "required": ["node_ids", "reason"],
            "additionalProperties": False,
        }

    @staticmethod
    def _verifier_user_prompt(chunk_text: str, candidates: List[Candidate]) -> str:
        lines = ["Candidate parameters:"]
        for c in candidates:
            lines.append(f"- id: {c.node} | description: {c.description}")
        lines.append("")
        lines.append("Passage:")
        lines.append(chunk_text.strip())
        lines.append("")
        lines.append(
            'Respond as JSON: {"node_ids": [<ids above that are genuinely discussed; '
            '[] if none>], "reason": <short justification>}.'
        )
        return "\n".join(lines)

    async def verify(
        self,
        chunk_text: str,
        candidates: List[Candidate],
        llm,
        calling_agent: Optional[str] = None,
    ) -> List[Candidate]:
        """Confirm which shortlisted nodes the chunk genuinely discusses.

        Args:
            llm: an object exposing `agen(messages, response_schema=..., calling_agent=...)`
                 (e.g. qmix_report_writer.llm.ollama_chat.OllamaChat).

        Returns the confirmed candidates (subset of the input), preserving the
        shortlist's score order. Empty list if the LLM rejects all of them.
        """
        if not candidates:
            return []
        messages = [
            Message(role="system", content=_VERIFIER_SYSTEM),
            Message(role="user", content=self._verifier_user_prompt(chunk_text, candidates)),
        ]
        raw = await llm.agen(
            messages,
            response_schema=self._verifier_schema(candidates),
            calling_agent=calling_agent,
        )
        data = _loads_lenient(raw)
        if not isinstance(data, dict):
            return []
        ids = data.get("node_ids")
        if not isinstance(ids, list):
            return []
        confirmed = {nid for nid in ids if isinstance(nid, str)}
        # Preserve shortlist (score) order so downstream priority is stable.
        return [c for c in candidates if c.node in confirmed]

    async def amatch(
        self,
        chunk_text: str,
        llm,
        top_k: Optional[int] = None,
        calling_agent: Optional[str] = None,
    ) -> List[Candidate]:
        """Full pipeline for one chunk: shortlist then verify. Returns confirmed candidates."""
        candidates = self.shortlist(chunk_text, top_k=top_k)
        if not candidates:
            return []
        return await self.verify(chunk_text, candidates, llm, calling_agent=calling_agent)

    async def expand(
        self,
        chunk_text: str,
        llm,
        k: Optional[int] = None,
        top_k: Optional[int] = None,
        max_neighbors: Optional[int] = None,
        calling_agent: Optional[str] = None,
    ) -> Dict[str, object]:
        """Match a chunk and return the k-hop neighbourhood of every confirmed node.

        Each matched node is expanded into its sources/effects; the combined
        neighbours are ranked so that neighbours of the *better-matching* nodes (and
        nearer hops) come first, deduplicated by (node, direction), and truncated to
        `max_neighbors`. Duplicate-description matches (e.g. `..._F20` and `..._F94`)
        are NOT deduplicated as matches — they tend to share neighbours, so the
        per-neighbour dedup and truncation handle the overlap.

        Returns:
            {
                "matched_nodes": [{"node", "description", "score"}, ...],  # score order
                "neighbors":     [ConnectedNode, ...],                    # ranked, truncated
                "k":             <hops used>,
            }
            `neighbors` is empty when no node is confirmed.
        """
        matched = await self.amatch(chunk_text, llm, top_k=top_k, calling_agent=calling_agent)
        k_val = self.manager.default_k if k is None else k
        cap = self.max_neighbors if max_neighbors is None else max_neighbors

        matched_names = {c.node for c in matched}
        collected: List[ConnectedNode] = []
        for cand in matched:
            for cn in self.manager.connected_nodes(cand.node, k=k_val, direction="both"):
                if cn.node in matched_names:
                    continue  # a confirmed node is a primary topic, not a neighbour
                cn.origin = cand.node
                cn.match_score = cand.score
                collected.append(cn)

        # Priority: neighbours of better-matching nodes first, then nearer hops.
        collected.sort(key=lambda c: (-c.match_score, c.hops, c.node))
        seen = set()
        ranked: List[ConnectedNode] = []
        for cn in collected:
            key = (cn.node, cn.direction)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(cn)
        if cap is not None:
            ranked = ranked[:cap]

        return {
            "matched_nodes": [
                {"node": c.node, "description": c.description, "score": c.score} for c in matched
            ],
            "neighbors": ranked,
            "k": k_val,
        }
