"""
Adaptation layer between the report-writer pipeline and the trimmed PBDS core.

The PBDS core (`pdbs_pareto_core.py`) knows how to read the LEANDREA parameter
workbook and build a directed dependency graph whose edges carry the connecting
Excel formula(s). This module wraps that graph for the pipeline's actual use
case: given a parameter node, return every node reachable within k hops together
with the formula(s) that connect them, split into

  * sources  — upstream nodes the origin is *computed from* (its causes), and
  * effects  — downstream nodes the origin *feeds into* (its consequences),

so the caller can go back to the RAG database and pull documents about the
related parameters, producing a report that also covers sources and effects.

Scope note: this layer assumes it is handed an EXACT node name. Resolving a free
text RAG chunk to a node name (fuzzy/alias matching) is deliberately left for a
later step; `nodes()` is exposed so that step has the canonical names to match
against.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from .pdbs_pareto_core import build_dependency_graph_from_main, load_main_parameters

# Traversal directions accepted by the query methods.
_UPSTREAM = "sources"     # predecessors: nodes the origin is computed from
_DOWNSTREAM = "effects"   # successors: nodes the origin feeds into
_BOTH = "both"


@dataclass
class ConnectedNode:
    """A node reached from an origin node, with how it is connected.

    Attributes:
        node:      the reachable parameter name.
        direction: "sources" if upstream of the origin, "effects" if downstream.
        hops:      shortest number of dependency edges between origin and node.
        path:      node sequence from the origin to this node (inclusive of both).
        formulas:  the formula(s) on the single edge directly linking this node to
                   its predecessor on `path` — i.e. the immediate connection.
        path_formulas: the formula(s) for every edge along `path`, in order, so the
                   full chain is available for multi-hop connections.
        owner:     workbook owner of the node (from the graph node attributes).
        row:       source row of the node in the main sheet.
        row_type:  "bounded_input" or "calculated_parameter".
        origin:    when produced by NodeMatcher.expand, the matched node this was
                   reached from; None otherwise.
        match_score: BM25 score of that matched origin node (0.0 if unset).
    """

    node: str
    direction: str
    hops: int
    path: List[str]
    formulas: List[str]
    path_formulas: List[List[str]] = field(default_factory=list)
    owner: str = ""
    row: Optional[int] = None
    row_type: str = ""
    origin: Optional[str] = None
    match_score: float = 0.0


class PBDSManager:
    """Build the dependency graph once and answer k-hop neighbourhood queries.

    The graph is built lazily on first use and cached. Construction reads the
    workbook (openpyxl) and parses formulas, so it is not free; reuse one manager
    instance across queries.
    """

    def __init__(self, workbook_path: str, default_k: int = 1):
        """
        Args:
            workbook_path: path to the LEANDREA parameter workbook (.xlsx).
            default_k:     hop radius used when a query does not specify one.
        """
        self.workbook_path = workbook_path
        self.default_k = default_k
        self._graph: Optional[nx.DiGraph] = None
        self._descriptions: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------ #
    # Graph access (lazy)
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        """Build the graph and the node->description map together (one workbook read)."""
        if self._graph is not None:
            return
        params_df = load_main_parameters(self.workbook_path)
        descriptions: Dict[str, str] = {}
        for _, r in params_df.iterrows():
            name = str(r["Parameter"])
            desc = r["Description"]
            descriptions[name] = desc.strip() if isinstance(desc, str) else ""
        self._descriptions = descriptions
        self._graph = build_dependency_graph_from_main(self.workbook_path, params_df)

    @property
    def graph(self) -> nx.DiGraph:
        """The cached dependency graph, built on first access."""
        self._ensure_loaded()
        assert self._graph is not None
        return self._graph

    def nodes(self) -> List[str]:
        """All node (parameter) names. Used by the later text->node matching step."""
        return list(self.graph.nodes)

    def descriptions(self) -> Dict[str, str]:
        """Map of node name -> human-readable description (the text-matching bridge).

        The description is the workbook's column-C label (e.g. "Pellet diameter");
        empty string when the row has no description. Node names themselves are
        machine-y (e.g. "Pellet_diameter_F8") and rarely appear verbatim in prose,
        so matching is done against these descriptions.
        """
        self._ensure_loaded()
        assert self._descriptions is not None
        return self._descriptions

    def has_node(self, node: str) -> bool:
        """True if `node` is an exact node name in the graph."""
        return node in self.graph

    # ------------------------------------------------------------------ #
    # k-hop queries
    # ------------------------------------------------------------------ #
    def connected_nodes(
        self,
        node: str,
        k: Optional[int] = None,
        direction: str = _BOTH,
    ) -> List[ConnectedNode]:
        """Return nodes reachable from `node` within `k` hops.

        Args:
            node:      exact node name (must already exist in the graph).
            k:         hop radius; defaults to `self.default_k`. Must be >= 1.
            direction: "sources" (upstream only), "effects" (downstream only) or
                       "both" (default).

        Returns:
            A list of ConnectedNode, sorted by (hops, direction, node). When
            direction is "both", a node reachable both up- and downstream appears
            once per direction.

        Raises:
            KeyError:   if `node` is not in the graph.
            ValueError: if `direction` is invalid or `k` < 1.
        """
        if node not in self.graph:
            raise KeyError(f"{node!r} is not a node in the dependency graph.")
        k = self.default_k if k is None else k
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if direction not in (_UPSTREAM, _DOWNSTREAM, _BOTH):
            raise ValueError(
                f"direction must be one of {_UPSTREAM!r}, {_DOWNSTREAM!r}, {_BOTH!r}; got {direction!r}."
            )

        results: List[ConnectedNode] = []
        if direction in (_UPSTREAM, _BOTH):
            results.extend(self._bfs(node, k, downstream=False))
        if direction in (_DOWNSTREAM, _BOTH):
            results.extend(self._bfs(node, k, downstream=True))
        results.sort(key=lambda c: (c.hops, c.direction, c.node))
        return results

    def neighborhood(
        self,
        node: str,
        k: Optional[int] = None,
    ) -> Dict[str, object]:
        """Convenience wrapper grouping the k-hop result into sources and effects.

        Returns a dict:
            {
                "node": <origin>,
                "k": <hops>,
                "sources": [ConnectedNode, ...],   # upstream / causes
                "effects": [ConnectedNode, ...],   # downstream / consequences
            }
        """
        k = self.default_k if k is None else k
        connected = self.connected_nodes(node, k=k, direction=_BOTH)
        return {
            "node": node,
            "k": k,
            "sources": [c for c in connected if c.direction == _UPSTREAM],
            "effects": [c for c in connected if c.direction == _DOWNSTREAM],
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _bfs(self, origin: str, k: int, downstream: bool) -> List[ConnectedNode]:
        """Breadth-first traversal up to depth `k` in one direction.

        downstream=True follows successors (effects); downstream=False follows
        predecessors (sources). Edge formulas are read in the graph's native
        orientation (src -> param), independent of traversal direction.
        """
        G = self.graph
        direction = _DOWNSTREAM if downstream else _UPSTREAM

        depth: Dict[str, int] = {origin: 0}
        parent: Dict[str, Optional[str]] = {origin: None}
        edge_formulas: Dict[str, List[str]] = {origin: []}

        queue: deque = deque([origin])
        reached: List[str] = []
        while queue:
            u = queue.popleft()
            if depth[u] >= k:
                continue
            neighbors = G.successors(u) if downstream else G.predecessors(u)
            for v in neighbors:
                if v in depth:
                    continue
                depth[v] = depth[u] + 1
                parent[v] = u
                # Edge orientation in the graph is always source -> dependent.
                edge = (u, v) if downstream else (v, u)
                edge_formulas[v] = list(G.edges[edge].get("formulas", []))
                queue.append(v)
                reached.append(v)

        results: List[ConnectedNode] = []
        for v in reached:
            path = self._reconstruct_path(v, parent)
            path_formulas = [edge_formulas[n] for n in path[1:]]
            attrs = G.nodes[v]
            results.append(
                ConnectedNode(
                    node=v,
                    direction=direction,
                    hops=depth[v],
                    path=path,
                    formulas=edge_formulas[v],
                    path_formulas=path_formulas,
                    owner=attrs.get("owner", ""),
                    row=attrs.get("row"),
                    row_type=attrs.get("row_type", ""),
                )
            )
        return results

    @staticmethod
    def _reconstruct_path(node: str, parent: Dict[str, Optional[str]]) -> List[str]:
        """Build the origin->node path from a BFS parent map."""
        path = [node]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        path.reverse()
        return path
