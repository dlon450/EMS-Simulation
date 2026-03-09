# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True

from __future__ import annotations

from array import array
from collections import OrderedDict
import math

cimport cython
from libc.math cimport INFINITY


@cython.final
cdef class ShortestPathCache:
    """Cython-accelerated Dijkstra shortest-path cache for a fixed directed graph.

    This class mirrors the Python ``src.pathfinding.ShortestPathCache`` API so it can
    be used as a drop-in replacement by importing ``src.pathfinding``.
    """

    cdef public object graph
    cdef public int max_cache_entries
    cdef public long cache_hits
    cdef public long cache_misses

    cdef object _cache  # OrderedDict[(mode, source)] -> (dist, prev_arc)
    cdef object _mode_arc_times  # mode_index -> array('d')

    cdef int _num_nodes
    cdef int _num_arcs
    cdef int _heap_capacity

    cdef object _arc_from_arr_obj
    cdef object _arc_to_arr_obj
    cdef int[:] _arc_from
    cdef int[:] _arc_to

    # CSR outgoing-arc structure for faster adjacency traversal.
    cdef object _out_offsets_arr_obj
    cdef object _out_arc_indices_arr_obj
    cdef int[:] _out_offsets
    cdef int[:] _out_arc_indices

    def __init__(self, graph, max_cache_entries: int = 128):
        cdef int m
        cdef int n
        cdef int i
        cdef int u
        cdef int total_out
        cdef int write_i
        cdef object arc
        cdef object out

        self.graph = graph
        self.max_cache_entries = max_cache_entries
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache = OrderedDict()
        self._mode_arc_times = {}

        if not self.graph.nodes or self.graph.nodes[0].index != 0:
            raise ValueError("Graph must be 1-based with a dummy node at index 0")
        if not self.graph.arcs or self.graph.arcs[0].index != 0:
            raise ValueError("Graph must be 1-based with a dummy arc at index 0")
        if not self.graph.out_arcs:
            raise ValueError("Graph adjacency is missing; call graph.build_adjacency() first")

        n = len(self.graph.nodes) - 1
        m = len(self.graph.arcs) - 1
        self._num_nodes = n
        self._num_arcs = m

        # arc endpoint arrays (1-based; index 0 unused)
        self._arc_from_arr_obj = array('i', [0]) * (m + 1)
        self._arc_to_arr_obj = array('i', [0]) * (m + 1)
        self._arc_from = self._arc_from_arr_obj
        self._arc_to = self._arc_to_arr_obj

        for i in range(1, m + 1):
            arc = self.graph.arcs[i]
            if arc.from_node_index is None or arc.to_node_index is None:
                raise ValueError(f"Arc {i} is missing endpoints")
            self._arc_from[i] = <int>arc.from_node_index
            self._arc_to[i] = <int>arc.to_node_index

        # Build compact outgoing adjacency over arc indices.
        self._out_offsets_arr_obj = array('i', [0]) * (n + 2)
        self._out_offsets = self._out_offsets_arr_obj

        total_out = 0
        for u in range(1, n + 1):
            self._out_offsets[u] = total_out
            out = self.graph.out_arcs[u]
            total_out += len(out)
        self._out_offsets[n + 1] = total_out

        self._out_arc_indices_arr_obj = array('i', [0]) * total_out
        self._out_arc_indices = self._out_arc_indices_arr_obj

        write_i = 0
        for u in range(1, n + 1):
            out = self.graph.out_arcs[u]
            for i in range(len(out)):
                self._out_arc_indices[write_i] = <int>out[i]
                write_i += 1

        # Dijkstra with duplicate pushes has <= O(E) pushes in practice;
        # allocate generously to avoid reallocation in the hot loop.
        self._heap_capacity = max(32, (m * 4) + n + 16)

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_arcs(self) -> int:
        return self._num_arcs

    def clear(self) -> None:
        """Clear all cached shortest path trees."""
        self._cache.clear()
        self._mode_arc_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    cdef void _touch(self, tuple key, tuple value):
        self._cache[key] = value
        self._cache.move_to_end(key, last=True)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    cdef object _ensure_mode_arc_times(self, int mode_index, object arc_times):
        cdef object arr = self._mode_arc_times.get(mode_index)
        if arr is None:
            arr = array('d', arc_times)
            self._mode_arc_times[mode_index] = arr
        return arr

    cpdef tuple get_tree(self, int mode_index, int source, object arc_times):
        """Return (dist, prev_arc) arrays for a given (mode, source).

        The arrays are 1-based with index 0 unused.
        """

        cdef tuple key
        cdef object cached
        cdef object mode_arc_times
        cdef tuple result

        if mode_index < 1:
            raise ValueError("mode_index must be >= 1")
        if source < 1 or source > self._num_nodes:
            raise ValueError(f"source must be in 1..{self._num_nodes}, got {source}")
        if len(arc_times) != self._num_arcs + 1:
            raise ValueError(
                f"arc_times length mismatch: expected {self._num_arcs + 1}, got {len(arc_times)}"
            )

        key = (mode_index, source)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key, last=True)
            return cached

        self.cache_misses += 1
        mode_arc_times = self._ensure_mode_arc_times(mode_index, arc_times)
        result = self._dijkstra_all(source, mode_arc_times)
        self._touch(key, result)
        return result

    cpdef double shortest_time(self, int mode_index, int source, int target, object arc_times):
        """Shortest travel time from source to target for a given travel mode."""

        cdef tuple tree
        cdef object dist

        if target < 1 or target > self._num_nodes:
            raise ValueError(f"target must be in 1..{self._num_nodes}, got {target}")

        tree = self.get_tree(mode_index, source, arc_times)
        dist = tree[0]
        return <double>dist[target]

    cpdef list shortest_path_arcs(
        self,
        int mode_index,
        int source,
        int target,
        object arc_times,
    ):
        """Return the shortest path as a list of arc indices."""

        cdef tuple tree
        cdef object dist
        cdef object prev_arc
        cdef list path
        cdef int v
        cdef int a

        if source == target:
            return []

        tree = self.get_tree(mode_index, source, arc_times)
        dist = tree[0]
        prev_arc = tree[1]

        if not math.isfinite(<double>dist[target]):
            raise ValueError(f"No path from {source} to {target} in mode {mode_index}")

        path = []
        v = target
        while v != source:
            a = <int>prev_arc[v]
            if a == 0:
                raise ValueError(f"Broken predecessor chain from {source} to {target}")
            path.append(a)
            v = self._arc_from[a]

        path.reverse()
        return path

    cpdef list shortest_path_nodes(
        self,
        int mode_index,
        int source,
        int target,
        object arc_times,
    ):
        """Return the shortest path as a list of node indices."""

        cdef list arcs
        cdef list nodes
        cdef int i
        cdef int a

        arcs = self.shortest_path_arcs(mode_index, source, target, arc_times)
        nodes = [source]

        for i in range(len(arcs)):
            a = <int>arcs[i]
            nodes.append(self._arc_to[a])

        return nodes

    cdef tuple _dijkstra_all(self, int source, object arc_times_obj):
        """Compute shortest paths from ``source`` to all nodes.

        Returns
        -------
        (dist, prev_arc)
            ``dist`` and ``prev_arc`` are array('d') / array('i') with 1-based indexing.
        """

        cdef int n = self._num_nodes
        cdef int heap_capacity = self._heap_capacity

        cdef object dist_arr_obj = array('d', [INFINITY]) * (n + 1)
        cdef object prev_arr_obj = array('i', [0]) * (n + 1)
        cdef double[:] dist = dist_arr_obj
        cdef int[:] prev_arc = prev_arr_obj

        cdef object heap_dist_arr_obj = array('d', [0.0]) * (heap_capacity + 1)
        cdef object heap_node_arr_obj = array('i', [0]) * (heap_capacity + 1)
        cdef double[:] heap_dist = heap_dist_arr_obj
        cdef int[:] heap_node = heap_node_arr_obj

        cdef double[:] arc_times = arc_times_obj

        cdef int heap_size = 1
        cdef int i
        cdef int parent
        cdef int left
        cdef int right
        cdef int child
        cdef int u
        cdef int v
        cdef int a
        cdef int k
        cdef int start_i
        cdef int end_i

        cdef double d_u
        cdef double w
        cdef double nd
        cdef double last_d
        cdef int last_node

        dist[source] = 0.0
        heap_dist[1] = 0.0
        heap_node[1] = source

        while heap_size > 0:
            # Pop min
            d_u = heap_dist[1]
            u = heap_node[1]

            last_d = heap_dist[heap_size]
            last_node = heap_node[heap_size]
            heap_size -= 1

            if heap_size > 0:
                i = 1
                while True:
                    left = i << 1
                    if left > heap_size:
                        break
                    right = left + 1
                    child = left
                    if right <= heap_size and heap_dist[right] < heap_dist[left]:
                        child = right
                    if heap_dist[child] >= last_d:
                        break
                    heap_dist[i] = heap_dist[child]
                    heap_node[i] = heap_node[child]
                    i = child
                heap_dist[i] = last_d
                heap_node[i] = last_node

            # Ignore stale queue entries
            if d_u > dist[u]:
                continue

            # Relax outgoing arcs for u
            start_i = self._out_offsets[u]
            end_i = self._out_offsets[u + 1]
            for k in range(start_i, end_i):
                a = self._out_arc_indices[k]
                v = self._arc_to[a]
                w = arc_times[a]

                if w <= 0:
                    raise ValueError(f"Arc travel time must be > 0 (mode time for arc {a} was {w})")

                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev_arc[v] = a

                    heap_size += 1
                    if heap_size > heap_capacity:
                        raise MemoryError(
                            "Internal Dijkstra heap capacity exceeded; "
                            "increase allocation in pathfinding_cy.ShortestPathCache"
                        )

                    i = heap_size
                    while i > 1:
                        parent = i >> 1
                        if heap_dist[parent] <= nd:
                            break
                        heap_dist[i] = heap_dist[parent]
                        heap_node[i] = heap_node[parent]
                        i = parent
                    heap_dist[i] = nd
                    heap_node[i] = v

        return (dist_arr_obj, prev_arr_obj)
