from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .newick import Node, ancestors, distance_to_ancestor, edge_depth, leaf_map, mrca


@dataclass(frozen=True)
class TripletObservation:
    taxa: tuple[str, str, str]
    topology_index: int
    topology_label: str
    internal_branch_length: float


@dataclass(frozen=True)
class TripletBatchPlan:
    combinations: tuple[tuple[str, str, str], ...]
    pairs: tuple[tuple[str, str], ...]
    pair_indices: np.ndarray

    @classmethod
    def from_combinations(
        cls, combinations: list[tuple[str, str, str]]
    ) -> "TripletBatchPlan":
        normalized = tuple(tuple(sorted(combination)) for combination in combinations)
        pairs = tuple(sorted({pair for combination in normalized for pair in _pairs(combination)}))
        mapping = {pair: index for index, pair in enumerate(pairs)}
        pair_indices = np.asarray(
            [[mapping[pair] for pair in _pairs(combination)] for combination in normalized],
            dtype=int,
        )
        return cls(normalized, pairs, pair_indices)


class TripletTreeIndex:
    """One-tree ancestry index for repeated rooted-triplet extraction."""

    def __init__(self, root: Node):
        self.root = root
        self.mapping = leaf_map(root)
        self.depth: dict[Node, int] = {}
        self.root_distance: dict[Node, float] = {}
        self.ancestor_paths = {name: ancestors(node) for name, node in self.mapping.items()}
        self.ancestor_sets = {name: set(path) for name, path in self.ancestor_paths.items()}
        self.pair_cache: dict[tuple[str, str], Node] = {}

        def visit(node: Node, depth: int, distance: float) -> None:
            self.depth[node] = depth
            self.root_distance[node] = distance
            for child in node.children:
                if child.length is None and not child.is_leaf:
                    raise ValueError("Branch length is required for triplet features")
                edge_length = 0.0 if child.length is None else float(child.length)
                visit(child, depth + 1, distance + edge_length)

        visit(root, 0, 0.0)

    def pair_mrca(self, first: str, second: str) -> Node:
        key = tuple(sorted((first, second)))
        if key not in self.pair_cache:
            if first not in self.mapping or second not in self.mapping:
                missing = [name for name in (first, second) if name not in self.mapping]
                raise ValueError(f"Missing taxa: {missing}")
            second_ancestors = self.ancestor_sets[second]
            for candidate in self.ancestor_paths[first]:
                if candidate in second_ancestors:
                    self.pair_cache[key] = candidate
                    break
            else:
                raise ValueError("Tree is disconnected")
        return self.pair_cache[key]

    def extract(self, taxa: tuple[str, str, str]) -> TripletObservation:
        taxa = tuple(sorted(taxa))
        missing = [name for name in taxa if name not in self.mapping]
        if missing:
            raise ValueError(f"Missing taxa: {missing}")
        pair_nodes = [self.pair_mrca(first, second) for first, second in _pairs(taxa)]
        depths = [self.depth[node] for node in pair_nodes]
        best_depth = max(depths)
        best = [index for index, depth in enumerate(depths) if depth == best_depth]
        root_depth = min(depths)
        if len(best) != 1 or best_depth == root_depth:
            raise ValueError(f"Unresolved rooted triplet for taxa {taxa}")
        index = best[0]
        triplet_root = min(pair_nodes, key=lambda node: self.depth[node])
        length = self.root_distance[pair_nodes[index]] - self.root_distance[triplet_root]
        return TripletObservation(taxa, index, topology_labels(taxa)[index], float(length))

    def extract_many(self, plan: TripletBatchPlan) -> tuple[np.ndarray, np.ndarray]:
        """Vectorize topology and branch-length extraction for fixed triplets."""

        missing = sorted(
            {taxon for combination in plan.combinations for taxon in combination} - set(self.mapping)
        )
        if missing:
            raise ValueError(f"Missing taxa: {missing}")
        pair_nodes = [self.pair_mrca(first, second) for first, second in plan.pairs]
        pair_depths = np.asarray([self.depth[node] for node in pair_nodes], dtype=int)
        pair_distances = np.asarray([self.root_distance[node] for node in pair_nodes], dtype=float)
        depths = pair_depths[plan.pair_indices]
        best = depths.argmax(axis=1)
        best_depth = depths.max(axis=1)
        root_depth = depths.min(axis=1)
        unique_best = (depths == best_depth[:, None]).sum(axis=1) == 1
        resolved = unique_best & (best_depth > root_depth)
        if not resolved.all():
            first = int(np.flatnonzero(~resolved)[0])
            raise ValueError(f"Unresolved rooted triplet for taxa {plan.combinations[first]}")
        roots = depths.argmin(axis=1)
        rows = np.arange(len(plan.combinations))
        lengths = (
            pair_distances[plan.pair_indices[rows, best]]
            - pair_distances[plan.pair_indices[rows, roots]]
        )
        return best.astype(np.int8), lengths


def triplet_combinations(taxa: list[str] | tuple[str, ...]) -> list[tuple[str, str, str]]:
    return list(itertools.combinations(sorted(taxa), 3))


def topology_labels(taxa: tuple[str, str, str]) -> tuple[str, str, str]:
    a, b, c = taxa
    return (f"{a}{b}|{c}", f"{a}{c}|{b}", f"{b}{c}|{a}")


def _pairs(taxa: tuple[str, str, str]):
    a, b, c = taxa
    return ((a, b), (a, c), (b, c))


def extract_triplet(root: Node, taxa: tuple[str, str, str]) -> TripletObservation:
    taxa = tuple(sorted(taxa))
    mapping = leaf_map(root)
    missing = [x for x in taxa if x not in mapping]
    if missing:
        raise ValueError(f"Missing taxa: {missing}")

    triplet_root = mrca([mapping[x] for x in taxa])
    pair_mrcas = [mrca([mapping[x], mapping[y]]) for x, y in _pairs(taxa)]
    depths = [edge_depth(node) for node in pair_mrcas]
    best_depth = max(depths)
    best = [i for i, depth in enumerate(depths) if depth == best_depth and pair_mrcas[i] is not triplet_root]
    if len(best) != 1:
        raise ValueError(f"Unresolved rooted triplet for taxa {taxa}")

    index = best[0]
    length = distance_to_ancestor(pair_mrcas[index], triplet_root)
    return TripletObservation(taxa, index, topology_labels(taxa)[index], length)


def true_triplet_labels(backbone: Node) -> dict[str, int]:
    taxa = sorted(x.name for x in backbone.leaves() if x.name is not None)
    result = {}
    for combo in triplet_combinations(taxa):
        obs = extract_triplet(backbone, combo)
        result[",".join(combo)] = obs.topology_index
    return result
