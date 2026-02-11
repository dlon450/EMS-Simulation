from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import heapq
import math
from collections import OrderedDict

from .network import Graph


CacheKey = Tuple[int, int]  # (mode_index, source_node)


@dataclass
class ShortestPathCache:
    """Dijkstra shortest-path cache for a fixed directed graph.

    Parameters
    ----------
    graph:
        Graph whose adjacency has been built (``graph.build_adjacency()``).
    max_cache_entries:
        Maximum number of (mode, source) entries cached at any time.

    Notes
    -----
    The cache stores full-distance arrays from the source to every node,
    along with predecessor arc indices for path reconstruction.
    """

    graph: Graph
    max_cache_entries: int = 128

    # arc endpoint arrays (1-based; index 0 unused)
    arc_from: List[int] = field(init=False)
    arc_to: List[int] = field(init=False)

    # LRU cache: (mode, source) -> (dist[0..n], prev_arc[0..n])
    _cache: "OrderedDict[CacheKey, Tuple[List[float], List[int]]]" = field(
        default_factory=OrderedDict, init=False
    )

    cache_hits: int = 0
    cache_misses: int = 0

    def __post_init__(self) -> None:
        if not self.graph.nodes or self.graph.nodes[0].index != 0:
            raise ValueError("Graph must be 1-based with a dummy node at index 0")
        if not self.graph.arcs or self.graph.arcs[0].index != 0:
            raise ValueError("Graph must be 1-based with a dummy arc at index 0")
        if not self.graph.out_arcs:
            raise ValueError("Graph adjacency is missing; call graph.build_adjacency() first")

        m = len(self.graph.arcs) - 1
        self.arc_from = [0] * (m + 1)
        self.arc_to = [0] * (m + 1)
        for i in range(1, m + 1):
            a = self.graph.arcs[i]
            if a.from_node_index is None or a.to_node_index is None:
                raise ValueError(f"Arc {i} is missing endpoints")
            self.arc_from[i] = int(a.from_node_index)
            self.arc_to[i] = int(a.to_node_index)

    @property
    def num_nodes(self) -> int:
        return len(self.graph.nodes) - 1

    @property
    def num_arcs(self) -> int:
        return len(self.graph.arcs) - 1

    def clear(self) -> None:
        """Clear all cached shortest path trees."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _touch(self, key: CacheKey, value: Tuple[List[float], List[int]]) -> None:
        """Insert/move an entry in the LRU cache."""
        self._cache[key] = value
        self._cache.move_to_end(key, last=True)
        # evict oldest
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    def get_tree(self, mode_index: int, source: int, arc_times: List[float]) -> Tuple[List[float], List[int]]:
        """Return (dist, prev_arc) arrays for a given (mode, source).

        The arrays are 1-based with index 0 unused.
        """
        if mode_index < 1:
            raise ValueError("mode_index must be >= 1")
        if not (1 <= source <= self.num_nodes):
            raise ValueError(f"source must be in 1..{self.num_nodes}, got {source}")
        if len(arc_times) != self.num_arcs + 1:
            raise ValueError(
                f"arc_times length mismatch: expected {self.num_arcs + 1}, got {len(arc_times)}"
            )

        key = (mode_index, source)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key, last=True)
            return cached

        self.cache_misses += 1
        dist, prev_arc = self._dijkstra_all(source, arc_times)
        self._touch(key, (dist, prev_arc))
        return dist, prev_arc

    def shortest_time(self, mode_index: int, source: int, target: int, arc_times: List[float]) -> float:
        """Shortest travel time from source to target for a given travel mode."""
        if not (1 <= target <= self.num_nodes):
            raise ValueError(f"target must be in 1..{self.num_nodes}, got {target}")
        dist, _prev = self.get_tree(mode_index, source, arc_times)
        return dist[target]

    def shortest_path_arcs(
        self,
        mode_index: int,
        source: int,
        target: int,
        arc_times: List[float],
    ) -> List[int]:
        """Return the shortest path as a list of arc indices.

        Raises
        ------
        ValueError
            If the target is unreachable from the source.
        """
        if source == target:
            return []
        dist, prev_arc = self.get_tree(mode_index, source, arc_times)
        if not math.isfinite(dist[target]):
            raise ValueError(f"No path from {source} to {target} in mode {mode_index}")

        path: List[int] = []
        v = target
        while v != source:
            a = prev_arc[v]
            if a == 0:
                # Should not happen if dist[target] is finite, but keep a safe check.
                raise ValueError(f"Broken predecessor chain from {source} to {target}")
            path.append(a)
            v = self.arc_from[a]
        path.reverse()
        return path

    def shortest_path_nodes(
        self,
        mode_index: int,
        source: int,
        target: int,
        arc_times: List[float],
    ) -> List[int]:
        """Return the shortest path as a list of node indices."""
        arcs = self.shortest_path_arcs(mode_index, source, target, arc_times)
        nodes: List[int] = [source]
        u = source
        for a in arcs:
            v = self.arc_to[a]
            nodes.append(v)
            u = v
        return nodes

    # ------------------------------------------------------------------
    # Internal Dijkstra implementation
    # ------------------------------------------------------------------

    def _dijkstra_all(self, source: int, arc_times: List[float]) -> Tuple[List[float], List[int]]:
        """Compute shortest paths from ``source`` to all nodes."""

        n = self.num_nodes

        dist = [math.inf] * (n + 1)
        prev_arc = [0] * (n + 1)

        dist[source] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, source)]

        out_arcs = self.graph.out_arcs

        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u != dist[u]:
                continue
            # relax outgoing arcs
            for a in out_arcs[u]:
                v = self.arc_to[a]
                w = arc_times[a]
                # w should be positive; if it's not, we'd break Dijkstra.
                if w <= 0:
                    raise ValueError(f"Arc travel time must be > 0 (mode time for arc {a} was {w})")
                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev_arc[v] = a
                    heapq.heappush(heap, (nd, v))

        return dist, prev_arc
