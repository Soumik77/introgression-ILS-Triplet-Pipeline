from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedTopology:
    taxa: tuple[str, str, str]
    paired: frozenset[str]
    outgroup: str
    weight: float


@dataclass(frozen=True)
class AssemblyNode:
    taxa: frozenset[str]
    children: tuple["AssemblyNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_newick(self, root: bool = True) -> str:
        if self.is_leaf:
            body = next(iter(self.taxa))
        else:
            body = "(" + ",".join(c.to_newick(False) for c in self.children) + ")"
        return body + (";" if root else "")


def probability_topologies(
    taxa: tuple[str, str, str], probabilities: np.ndarray | list[float]
) -> list[WeightedTopology]:
    a, b, c = tuple(sorted(taxa))
    definitions = [((a, b), c), ((a, c), b), ((b, c), a)]
    return [
        WeightedTopology((a, b, c), frozenset(pair), outgroup, float(weight))
        for (pair, outgroup), weight in zip(definitions, probabilities)
    ]


def split_score(left: set[str], right: set[str], topologies: list[WeightedTopology]) -> float:
    """Return mean signed support among triplets resolved by this split.

    Dividing by the number of crossing triplets prevents a candidate split from
    winning only because it resolves more triplets. Without this normalization,
    the recursive search can prefer a balanced but topologically incorrect split
    even when every input triplet has its true one-hot label.
    """
    if not left or not right or left & right:
        return float("-inf")
    score = 0.0
    crossing_triplets: set[tuple[str, str, str]] = set()
    for topology in topologies:
        a, b = tuple(topology.paired)
        c = topology.outgroup
        sides = [a in left, b in left, c in left]
        if all(sides) or not any(sides):
            continue  # deferred
        crossing_triplets.add(topology.taxa)
        pair_together = (a in left) == (b in left)
        score += topology.weight if pair_together else -topology.weight
    if not crossing_triplets:
        return float("-inf")
    return score / len(crossing_triplets)


def _topology_contribution(
    topology: WeightedTopology, left: set[str], flipped_taxon: str | None = None
) -> float:
    a, b = tuple(topology.paired)
    c = topology.outgroup

    def is_left(taxon: str) -> bool:
        value = taxon in left
        return not value if taxon == flipped_taxon else value

    sides = [is_left(a), is_left(b), is_left(c)]
    if all(sides) or not any(sides):
        return 0.0
    return topology.weight if sides[0] == sides[1] else -topology.weight


def _unique_bipartitions(taxa: set[str]):
    ordered = sorted(taxa)
    anchor = ordered[0]
    others = ordered[1:]
    for size in range(0, len(others)):
        for subset in itertools.combinations(others, size):
            left = {anchor, *subset}
            right = taxa - left
            if right:
                yield left, right


def exact_best_split(taxa: set[str], topologies: list[WeightedTopology]) -> tuple[set[str], set[str]]:
    candidates = list(_unique_bipartitions(taxa))
    if not candidates:
        raise ValueError("Cannot split fewer than two taxa")
    return max(
        candidates,
        key=lambda lr: (split_score(lr[0], lr[1], topologies), -abs(len(lr[0]) - len(lr[1])), sorted(lr[0])),
    )


def _pair_support(a: str, b: str, topologies: list[WeightedTopology]) -> float:
    return sum(t.weight for t in topologies if t.paired == frozenset((a, b)))


def _pair_support_index(topologies: list[WeightedTopology]) -> dict[frozenset[str], float]:
    result: dict[frozenset[str], float] = {}
    for topology in topologies:
        result[topology.paired] = result.get(topology.paired, 0.0) + topology.weight
    return result


def initial_split(taxa: set[str], topologies: list[WeightedTopology]) -> tuple[set[str], set[str]]:
    ordered = sorted(taxa)
    if len(ordered) == 2:
        return {ordered[0]}, {ordered[1]}
    supports = _pair_support_index(topologies)

    best_supported: tuple[float, tuple[str, str]] | None = None
    seed_pair: tuple[str, str] | None = None
    nonnegative = all(value >= 0.0 for value in supports.values())
    for pair in itertools.combinations(ordered, 2):
        key = frozenset(pair)
        if nonnegative and key not in supports:
            seed_pair = pair
            break
        candidate = (supports.get(key, 0.0), pair)
        if best_supported is None or candidate < best_supported:
            best_supported = candidate
    if seed_pair is None:
        if best_supported is None:
            raise ValueError("Cannot choose a seed pair")
        seed_pair = best_supported[1]
    seed_a, seed_b = seed_pair

    neighbors: dict[str, dict[str, float]] = {taxon: {} for taxon in ordered}
    for pair, value in supports.items():
        first, second = tuple(pair)
        if first in neighbors and second in neighbors:
            neighbors[first][second] = value
            neighbors[second][first] = value
    left, right = {seed_a}, {seed_b}
    for taxon in [x for x in ordered if x not in {seed_a, seed_b}]:
        taxon_neighbors = neighbors[taxon]
        left_affinity = sum(value for other, value in taxon_neighbors.items() if other in left) / len(left)
        right_affinity = sum(value for other, value in taxon_neighbors.items() if other in right) / len(right)
        if left_affinity > right_affinity:
            left.add(taxon)
        elif right_affinity > left_affinity:
            right.add(taxon)
        elif len(left) <= len(right):
            left.add(taxon)
        else:
            right.add(taxon)
    return left, right


def _fm_pass(
    left: set[str], right: set[str], topologies: list[WeightedTopology]
) -> tuple[set[str], set[str], float]:
    start_left, start_right = set(left), set(right)
    start_score = split_score(left, right, topologies)
    ordered_taxa = sorted(left | right)
    taxon_index = {taxon: index for index, taxon in enumerate(ordered_taxa)}
    side = np.array([taxon in left for taxon in ordered_taxa], dtype=bool)
    unlocked = np.ones(len(ordered_taxa), dtype=bool)
    states: list[tuple[float, set[str], set[str]]] = []
    paired = [tuple(topology.paired) for topology in topologies]
    a = np.array([taxon_index[pair[0]] for pair in paired], dtype=int)
    b = np.array([taxon_index[pair[1]] for pair in paired], dtype=int)
    c = np.array([taxon_index[topology.outgroup] for topology in topologies], dtype=int)
    weights = np.array([topology.weight for topology in topologies], dtype=float)
    triplets = sorted({topology.taxa for topology in topologies})
    ta = np.array([taxon_index[triplet[0]] for triplet in triplets], dtype=int)
    tb = np.array([taxon_index[triplet[1]] for triplet in triplets], dtype=int)
    tc = np.array([taxon_index[triplet[2]] for triplet in triplets], dtype=int)

    def contributions(sa: np.ndarray, sb: np.ndarray, sc: np.ndarray) -> np.ndarray:
        deferred = (sa == sb) & (sb == sc)
        return np.where(deferred, 0.0, np.where(sa == sb, weights, -weights))

    def crossing(sa: np.ndarray, sb: np.ndarray, sc: np.ndarray) -> np.ndarray:
        return ~((sa == sb) & (sb == sc))

    sa, sb, sc = side[a], side[b], side[c]
    current_numerator = float(contributions(sa, sb, sc).sum())
    current_denominator = int(crossing(side[ta], side[tb], side[tc]).sum())
    current_score = (
        current_numerator / current_denominator
        if current_denominator
        else float("-inf")
    )

    while unlocked.any():
        sa, sb, sc = side[a], side[b], side[c]
        before = contributions(sa, sb, sc)
        gains = np.zeros(len(ordered_taxa), dtype=float)
        if len(topologies):
            np.add.at(gains, a, contributions(~sa, sb, sc) - before)
            np.add.at(gains, b, contributions(sa, ~sb, sc) - before)
            np.add.at(gains, c, contributions(sa, sb, ~sc) - before)

        tsa, tsb, tsc = side[ta], side[tb], side[tc]
        before_crossing = crossing(tsa, tsb, tsc)
        denominator_gains = np.zeros(len(ordered_taxa), dtype=int)
        if len(triplets):
            np.add.at(
                denominator_gains,
                ta,
                crossing(~tsa, tsb, tsc).astype(int) - before_crossing.astype(int),
            )
            np.add.at(
                denominator_gains,
                tb,
                crossing(tsa, ~tsb, tsc).astype(int) - before_crossing.astype(int),
            )
            np.add.at(
                denominator_gains,
                tc,
                crossing(tsa, tsb, ~tsc).astype(int) - before_crossing.astype(int),
            )

        left_count = int(side.sum())
        right_count = len(side) - left_count
        movable = np.flatnonzero(
            unlocked & np.where(side, left_count > 1, right_count > 1)
        )
        if not len(movable):
            break
        candidate_scores = np.full(len(ordered_taxa), float("-inf"), dtype=float)
        candidate_denominators = current_denominator + denominator_gains
        valid = candidate_denominators > 0
        candidate_scores[valid] = (
            current_numerator + gains[valid]
        ) / candidate_denominators[valid]
        valid_movable = [index for index in movable if valid[index]]
        if not valid_movable:
            break
        chosen = max(
            valid_movable,
            key=lambda index: (
                candidate_scores[index],
                gains[index],
                ordered_taxa[index],
            ),
        )
        side[chosen] = not side[chosen]
        current_numerator += float(gains[chosen])
        current_denominator += int(denominator_gains[chosen])
        current_score = current_numerator / current_denominator
        unlocked[chosen] = False
        state_left = {taxon for index, taxon in enumerate(ordered_taxa) if side[index]}
        states.append((current_score, state_left, set(ordered_taxa) - state_left))
    if not states:
        return start_left, start_right, 0.0
    best_score, best_left, best_right = max(states, key=lambda item: item[0])
    improvement = best_score - start_score
    if improvement > 1e-12:
        return best_left, best_right, improvement
    return start_left, start_right, 0.0


def fm_best_split(taxa: set[str], topologies: list[WeightedTopology]) -> tuple[set[str], set[str]]:
    if not topologies:
        ordered = sorted(taxa)
        midpoint = max(1, len(ordered) // 2)
        return set(ordered[:midpoint]), set(ordered[midpoint:])
    left, right = initial_split(taxa, topologies)
    while True:
        next_left, next_right, improvement = _fm_pass(left, right, topologies)
        if improvement <= 1e-12:
            return left, right
        left, right = next_left, next_right


def assemble_tree(
    taxa: set[str], topologies: list[WeightedTopology], method: str = "fm"
) -> AssemblyNode:
    taxa = set(taxa)
    if method == "exact" and len(taxa) > 12:
        raise ValueError(
            "Exact recursive bipartition search is restricted to at most 12 taxa. "
            "Use method='fm' for larger trees."
        )
    if len(taxa) == 1:
        return AssemblyNode(frozenset(taxa))
    if len(taxa) == 2:
        children = tuple(AssemblyNode(frozenset((x,))) for x in sorted(taxa))
        return AssemblyNode(frozenset(taxa), children)

    applicable = [t for t in topologies if set(t.taxa).issubset(taxa)]
    if method == "fm":
        left, right = fm_best_split(taxa, applicable)
    elif method == "exact":
        left, right = exact_best_split(taxa, applicable)
    else:
        raise ValueError("method must be 'fm' or 'exact'")

    left_topologies = [t for t in applicable if set(t.taxa).issubset(left)]
    right_topologies = [t for t in applicable if set(t.taxa).issubset(right)]
    children = (
        assemble_tree(left, left_topologies, method),
        assemble_tree(right, right_topologies, method),
    )
    children = tuple(sorted(children, key=lambda node: sorted(node.taxa)))
    return AssemblyNode(frozenset(taxa), children)
