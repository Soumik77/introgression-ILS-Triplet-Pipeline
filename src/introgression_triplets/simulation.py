from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import yaml

from .newick import Node, parse_newick


Topology: TypeAlias = str | tuple["Topology", "Topology"]


@dataclass(frozen=True)
class BackboneNode:
    name: str | None
    height: float
    children: tuple["BackboneNode", ...] = ()


def topology_key(topology: Topology) -> str:
    if isinstance(topology, str):
        return topology
    children = sorted((topology_key(topology[0]), topology_key(topology[1])))
    return f"({children[0]},{children[1]})"


def canonical_topology(left: Topology, right: Topology) -> Topology:
    return (left, right) if topology_key(left) <= topology_key(right) else (right, left)


@lru_cache(maxsize=None)
def enumerate_rooted_binary_topologies(taxa: tuple[str, ...]) -> tuple[Topology, ...]:
    """Enumerate every unordered rooted binary topology for labeled taxa."""

    taxa = tuple(sorted(taxa))
    if len(taxa) == 1:
        return (taxa[0],)
    anchor = taxa[0]
    remaining = taxa[1:]
    results: dict[str, Topology] = {}
    for extra_count in range(len(remaining)):
        for extra_left in itertools.combinations(remaining, extra_count):
            left_taxa = tuple(sorted((anchor, *extra_left)))
            right_taxa = tuple(x for x in taxa if x not in left_taxa)
            for left in enumerate_rooted_binary_topologies(left_taxa):
                for right in enumerate_rooted_binary_topologies(right_taxa):
                    topology = canonical_topology(left, right)
                    results[topology_key(topology)] = topology
    return tuple(results[key] for key in sorted(results))


def topology_shape(topology: Topology) -> str:
    if isinstance(topology, str):
        return "x"
    children = sorted((topology_shape(topology[0]), topology_shape(topology[1])))
    return f"({children[0]},{children[1]})"


def _node_to_topology(node: Node) -> Topology:
    if node.is_leaf:
        if node.name is None:
            raise ValueError("Backbone topology contains an unnamed leaf")
        return node.name
    if len(node.children) != 2:
        raise ValueError("Backbone must be rooted and binary")
    return canonical_topology(_node_to_topology(node.children[0]), _node_to_topology(node.children[1]))


def parse_topology(text: str) -> Topology:
    source = text.strip()
    if not source.endswith(";"):
        source += ";"
    return _node_to_topology(parse_newick(source))


def topology_leaf_count(topology: Topology) -> int:
    if isinstance(topology, str):
        return 1
    return topology_leaf_count(topology[0]) + topology_leaf_count(topology[1])


def colless_index(topology: Topology) -> int:
    """Return the Colless imbalance index for a rooted binary topology."""

    if isinstance(topology, str):
        return 0
    left_size = topology_leaf_count(topology[0])
    right_size = topology_leaf_count(topology[1])
    return abs(left_size - right_size) + colless_index(topology[0]) + colless_index(topology[1])


def normalized_colless_index(topology: Topology) -> float:
    n = topology_leaf_count(topology)
    maximum = (n - 1) * (n - 2) / 2
    return float(colless_index(topology) / maximum) if maximum else 0.0


def random_rooted_binary_topology(taxa: list[str], rng: random.Random) -> Topology:
    """Generate one deterministic-seed random rooted binary labeled topology."""

    nodes: list[Topology] = list(taxa)
    rng.shuffle(nodes)
    while len(nodes) > 1:
        first, second = sorted(rng.sample(range(len(nodes)), 2), reverse=True)
        left = nodes.pop(first)
        right = nodes.pop(second)
        nodes.append(canonical_topology(left, right))
    return nodes[0]


def _balance_stratified_random_topologies(
    taxa: list[str], count: int, rng: random.Random, candidate_multiplier: int
) -> list[Topology]:
    candidate_count = max(count * candidate_multiplier, count * 4)
    candidates: dict[str, Topology] = {}
    attempts = 0
    maximum_attempts = candidate_count * 20
    while len(candidates) < candidate_count and attempts < maximum_attempts:
        topology = random_rooted_binary_topology(taxa, rng)
        candidates[topology_key(topology)] = topology
        attempts += 1
    if len(candidates) < count:
        raise RuntimeError(f"Generated only {len(candidates)} unique backbones; requested {count}")

    ordered = sorted(candidates.values(), key=lambda value: (normalized_colless_index(value), topology_key(value)))
    quartiles: list[list[Topology]] = [[], [], [], []]
    for index, topology in enumerate(ordered):
        quartile = min(3, (4 * index) // len(ordered))
        quartiles[quartile].append(topology)
    for bucket in quartiles:
        rng.shuffle(bucket)

    targets = [count // 4 + (1 if index < count % 4 else 0) for index in range(4)]
    selected = [topology for bucket, target in zip(quartiles, targets) for topology in bucket[:target]]
    if len(selected) != count:
        raise RuntimeError("Could not construct the requested balance-stratified backbone sample")
    rng.shuffle(selected)
    return selected


def select_backbone_topologies(cfg: dict, taxa: list[str]) -> list[Topology]:
    strategy = str(cfg.get("backbone_strategy", "fixed"))
    if strategy == "fixed":
        source = cfg.get("backbone_topology", cfg.get("backbone_shape"))
        if not source:
            raise ValueError("Fixed strategy requires backbone_topology")
        topology = parse_topology(str(source))
        if sorted(_topology_taxa(topology)) != sorted(taxa):
            raise ValueError("Fixed backbone taxa do not match configured taxa")
        return [topology]
    if strategy == "explicit":
        sources = cfg.get("backbone_topologies", [])
        if not sources:
            raise ValueError("Explicit strategy requires backbone_topologies")
        parsed: dict[str, Topology] = {}
        for source in sources:
            topology = parse_topology(str(source))
            if sorted(_topology_taxa(topology)) != sorted(taxa):
                raise ValueError("Explicit backbone taxa do not match configured taxa")
            parsed[topology_key(topology)] = topology
        return [parsed[key] for key in sorted(parsed)]

    rng = random.Random(int(cfg["base_seed"]))

    if strategy == "balance_stratified_random":
        count = int(cfg.get("backbone_count", 40))
        if count < 4:
            raise ValueError("balance_stratified_random requires backbone_count >= 4")
        return _balance_stratified_random_topologies(
            taxa,
            count,
            rng,
            int(cfg.get("backbone_candidate_multiplier", 50)),
        )

    if len(taxa) > 8:
        raise ValueError(
            "Exhaustive topology enumeration is restricted to at most eight taxa. "
            "Use backbone_strategy: balance_stratified_random for larger taxon sets."
        )
    all_topologies = list(enumerate_rooted_binary_topologies(tuple(taxa)))
    count = int(cfg.get("backbone_count", len(all_topologies)))
    if not 1 <= count <= len(all_topologies):
        raise ValueError(f"backbone_count must be between 1 and {len(all_topologies)}")

    if strategy == "random":
        return rng.sample(all_topologies, count)
    if strategy == "all":
        return all_topologies[:count]
    if strategy != "stratified":
        raise ValueError(
            "backbone_strategy must be fixed, explicit, random, stratified, all, "
            "or balance_stratified_random"
        )

    buckets: dict[str, list[Topology]] = {}
    for topology in all_topologies:
        buckets.setdefault(topology_shape(topology), []).append(topology)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[Topology] = []
    shapes = sorted(buckets)
    while len(selected) < count:
        made_progress = False
        for shape in shapes:
            if buckets[shape] and len(selected) < count:
                selected.append(buckets[shape].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def _topology_taxa(topology: Topology) -> list[str]:
    if isinstance(topology, str):
        return [topology]
    return _topology_taxa(topology[0]) + _topology_taxa(topology[1])


def _max_internal_depth(topology: Topology, depth: int = 0) -> int:
    if isinstance(topology, str):
        return depth - 1
    return max(depth, _max_internal_depth(topology[0], depth + 1), _max_internal_depth(topology[1], depth + 1))


def backbone_from_topology(
    topology: Topology, internal_length: float, pendant_base: float = 1.0
) -> BackboneNode:
    """Create an ultrametric backbone with equal internal-edge lengths."""

    t = float(internal_length)
    p = float(pendant_base)
    max_depth = _max_internal_depth(topology)
    root_height = p + max_depth * t

    def build(current: Topology, depth: int) -> BackboneNode:
        if isinstance(current, str):
            return BackboneNode(current, 0.0)
        height = root_height - depth * t
        return BackboneNode(None, height, (build(current[0], depth + 1), build(current[1], depth + 1)))

    return build(topology, 0)


def six_taxon_backbone(internal_length: float, pendant_base: float = 1.0) -> BackboneNode:
    """Backward-compatible constructor for the original fixed backbone."""

    return backbone_from_topology(parse_topology("(((A,B),(C,D)),(E,F))"), internal_length, pendant_base)


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def render_backbone(root: BackboneNode) -> str:
    def render(node: BackboneNode, parent_height: float | None) -> str:
        if node.children:
            body = "(" + ",".join(render(c, node.height) for c in node.children) + ")"
        else:
            body = node.name or ""
        if parent_height is not None:
            body += ":" + _fmt(parent_height - node.height)
        return body

    return render(root, None) + ";"


def render_network(
    root: BackboneNode,
    donor: str | None,
    recipient: str | None,
    gamma: float,
    gene_flow_height: float,
) -> str:
    """Render a one-reticulation, time-consistent extended-Newick network.

    The minor edge goes donor -> recipient at a shared height above both tips.
    """

    if gamma <= 0.0:
        return render_backbone(root)
    if donor is None or recipient is None or donor == recipient:
        raise ValueError("Positive gamma requires distinct donor and recipient")
    if not 0.0 < gamma <= 0.5:
        raise ValueError("gamma must be in (0, 0.5]")

    seen = {n.name for n in _walk(root) if not n.children}
    if donor not in seen or recipient not in seen:
        raise ValueError("Donor and recipient must be tip taxa")

    h = float(gene_flow_height)

    def contains_taxon(node: BackboneNode, taxon: str) -> bool:
        return any(n.name == taxon for n in _walk(node) if not n.children)

    def render(node: BackboneNode, parent_height: float | None) -> str:
        if node.children:
            # Put the recipient-defining occurrence before the #H1 reference.
            ordered_children = sorted(
                node.children,
                key=lambda child: (
                    0 if contains_taxon(child, recipient) else 1 if contains_taxon(child, donor) else 2
                ),
            )
            body = "(" + ",".join(render(c, node.height) for c in ordered_children) + ")"
            if parent_height is not None:
                body += ":" + _fmt(parent_height - node.height)
            return body

        if parent_height is None or h <= 0.0 or h >= parent_height:
            raise ValueError("gene_flow_height must lie inside every affected pendant edge")
        name = node.name or ""
        remainder = parent_height - h
        if name == donor:
            return f"({name}:{_fmt(h)},#H1:0.0::{_fmt(gamma)}):{_fmt(remainder)}"
        if name == recipient:
            # PhyloNetworks infers the partner-edge inheritance as 1-gamma.
            return f"({name}:{_fmt(h)})#H1:{_fmt(remainder)}"
        return f"{name}:{_fmt(parent_height)}"

    return render(root, None) + ";"


def _walk(root: BackboneNode):
    yield root
    for child in root.children:
        yield from _walk(child)


def all_directions(taxa: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in taxa for b in taxa if a != b]


def parse_directions(value, taxa: list[str]) -> list[tuple[str, str]]:
    if value == "all":
        return all_directions(taxa)
    result = []
    for item in value:
        parts = str(item).split(">")
        if len(parts) != 2:
            raise ValueError(f"Invalid direction {item!r}; expected donor>recipient")
        donor, recipient = (p.strip() for p in parts)
        if donor not in taxa or recipient not in taxa or donor == recipient:
            raise ValueError(f"Invalid direction {item!r}")
        result.append((donor, recipient))
    return result


def _leaf_paths(topology: Topology, path: tuple[int, ...] = ()) -> dict[str, tuple[int, ...]]:
    if isinstance(topology, str):
        return {topology: path}
    return {
        **_leaf_paths(topology[0], path + (0,)),
        **_leaf_paths(topology[1], path + (1,)),
    }


def _path_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    common = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common += 1
    return len(first) + len(second) - 2 * common


def distance_stratified_directions(
    topology: Topology, base_seed: int, backbone_id: str
) -> list[tuple[str, str, str]]:
    """Select one close, one intermediate, and one distant directed tip pair."""

    paths = _leaf_paths(topology)
    pairs = sorted(
        itertools.combinations(sorted(paths), 2),
        key=lambda pair: (_path_distance(paths[pair[0]], paths[pair[1]]), pair),
    )
    if len(pairs) < 3:
        raise ValueError("Distance-stratified directions require at least three taxa")
    selected_indices = [0, len(pairs) // 2, len(pairs) - 1]
    pair_types = ["close", "intermediate", "distant"]
    result = []
    for pair_type, pair_index in zip(pair_types, selected_indices):
        first, second = pairs[pair_index]
        rng = random.Random(deterministic_seed(base_seed, f"{backbone_id}|{pair_type}"))
        donor, recipient = (first, second) if rng.random() < 0.5 else (second, first)
        result.append((donor, recipient, pair_type))
    return result


def configured_taxon_counts(cfg: dict) -> list[int]:
    """Return the requested taxon counts, including both range endpoints."""

    if "taxon_count_range" in cfg:
        specification = cfg["taxon_count_range"]
        start = int(specification["start"])
        end = int(specification["end"])
        step = int(specification.get("step", 1))
        if step <= 0 or end < start:
            raise ValueError("taxon_count_range requires end >= start and step > 0")
        counts = list(range(start, end + 1, step))
        if counts[-1] != end:
            raise ValueError("taxon_count_range step must land exactly on end")
    elif "taxon_counts" in cfg:
        counts = [int(value) for value in cfg["taxon_counts"]]
    elif "taxa" in cfg:
        counts = [len(cfg["taxa"])]
    else:
        counts = [int(cfg.get("taxon_count", 0))]
    if not counts or any(value < 3 for value in counts) or len(set(counts)) != len(counts):
        raise ValueError("Taxon counts must be unique integers of at least three")
    return counts


def configured_taxa(cfg: dict, count_override: int | None = None) -> list[str]:
    if "taxa" in cfg:
        if count_override is not None and count_override != len(cfg["taxa"]):
            raise ValueError("Explicit taxa cannot be combined with a different taxon count")
        taxa = [str(value) for value in cfg["taxa"]]
    else:
        count = int(count_override if count_override is not None else cfg.get("taxon_count", 0))
        if count < 3:
            raise ValueError("Specify at least three taxa using taxa or taxon_count")
        prefix = str(cfg.get("taxon_prefix", "T"))
        width = int(cfg.get("taxon_width", max(2, len(str(count)))))
        if width < 1:
            raise ValueError("taxon_width must be at least one")
        taxa = [f"{prefix}{index:0{width}d}" for index in range(1, count + 1)]
    if len(taxa) < 3 or len(set(taxa)) != len(taxa):
        raise ValueError("Taxon labels must contain at least three unique values")
    if any(any(character in label for character in "(),:;") for label in taxa):
        raise ValueError("Taxon labels cannot contain Newick punctuation")
    return taxa


def deterministic_seed(base_seed: int, key: str) -> int:
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") + int(base_seed)) % 2_147_483_647


def _triplet_count_for_size(cfg: dict, tree_size: int) -> int | None:
    if cfg.get("triplets_per_simulation") is not None:
        requested = int(cfg["triplets_per_simulation"])
    elif cfg.get("max_triplets_per_simulation") is not None:
        requested = int(cfg["max_triplets_per_simulation"])
    else:
        return None
    minimum = (tree_size + 2) // 3
    if requested < minimum:
        raise ValueError(
            f"Triplet cap {requested} cannot cover every taxon for tree_size={tree_size}; "
            f"minimum is {minimum}"
        )
    return min(math.comb(tree_size, 3), requested)


def _build_backbone_records(cfg: dict) -> list[dict]:
    taxon_counts = configured_taxon_counts(cfg)
    multi_size = len(taxon_counts) > 1
    if multi_size and str(cfg.get("backbone_assignment", "rotate")) != "cross_product":
        raise ValueError("Multi-size manifests require backbone_assignment: cross_product")

    records: list[dict] = []
    for tree_size in taxon_counts:
        taxa = configured_taxa(cfg, tree_size)
        local_cfg = dict(cfg)
        local_cfg["taxon_count"] = tree_size
        local_cfg["base_seed"] = (
            deterministic_seed(int(cfg["base_seed"]), f"backbones|tree_size={tree_size}")
            if multi_size
            else int(cfg["base_seed"])
        )
        if multi_size or "backbones_per_taxon_count" in cfg:
            local_cfg["backbone_count"] = int(
                cfg.get("backbones_per_taxon_count", cfg.get("backbone_count", 10))
            )
        topology_pool = select_backbone_topologies(local_cfg, taxa)
        balance_order = sorted(
            range(len(topology_pool)),
            key=lambda index: (
                normalized_colless_index(topology_pool[index]),
                topology_key(topology_pool[index]),
            ),
        )
        balance_bins: dict[int, str] = {}
        for rank, topology_index in enumerate(balance_order):
            balance_bins[topology_index] = f"Q{min(3, (4 * rank) // len(balance_order)) + 1}"

        size_prefix = f"n{tree_size:03d}_" if multi_size else ""
        for index, topology in enumerate(topology_pool, start=1):
            if len(topology_pool) == 1 and not multi_size:
                backbone_id = str(cfg.get("backbone_id", "backbone_fixed"))
            else:
                width = max(2 if multi_size else 3, len(str(len(topology_pool))))
                backbone_id = f"{size_prefix}backbone_{index:0{width}d}"
            records.append(
                {
                    "taxon_set_id": f"n{tree_size:03d}",
                    "tree_size": tree_size,
                    "triplets_per_simulation": _triplet_count_for_size(cfg, tree_size),
                    "taxa": taxa,
                    "backbone_id": backbone_id,
                    "backbone_shape": topology_key(topology),
                    "backbone_shape_class": topology_shape(topology),
                    "backbone_balance_score": normalized_colless_index(topology),
                    "backbone_balance_bin": balance_bins[index - 1],
                    "topology": topology,
                }
            )
    return records


def _replicates_for_gamma(cfg: dict, gamma: float) -> int:
    default = int(cfg.get("replicates", 1))
    key = "null_replicates" if gamma == 0.0 else "positive_replicates"
    count = int(cfg.get(key, default))
    if count < 1:
        raise ValueError(f"{key} must be at least one")
    return count


def select_balanced_analysis_manifest(
    manifest: pd.DataFrame, runs_per_backbone: int, base_seed: int
) -> pd.DataFrame:
    """Choose a deterministic, factor-balanced analysis subset per backbone.

    Selection is performed before observing simulated outcomes. The greedy
    objective matches each selected subset to the factor proportions in its
    candidate grid while avoiding duplicate scenario cells when possible.
    """

    if runs_per_backbone < 1:
        raise ValueError("runs_per_backbone must be at least one")
    required = {
        "backbone_id",
        "simulation_id",
        "scenario_group",
        "gamma",
        "ils_length",
        "gene_tree_count",
        "pair_type",
        "replicate",
        "tree_size",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing selection columns: {missing}")

    selected_indices: list[int] = []
    deadline_schedules: dict[int, list[tuple[str, list[int]]]] = {}
    factors = ["gamma", "ils_length", "gene_tree_count", "pair_type"]
    for backbone_id, group in manifest.groupby("backbone_id", sort=True):
        if len(group) < runs_per_backbone:
            raise ValueError(
                f"Backbone {backbone_id} has {len(group)} runs, fewer than {runs_per_backbone}"
            )
        gamma_levels = sorted(group["gamma"].astype(float).unique().tolist())
        ils_levels = sorted(group["ils_length"].astype(float).unique().tolist())
        gene_tree_levels = sorted(group["gene_tree_count"].astype(int).unique().tolist())
        positive_pair_types = sorted(
            value for value in group["pair_type"].astype(str).unique() if value != "none"
        )
        supervisor_grid = (
            runs_per_backbone in {3, 12}
            and len(gamma_levels) == 3
            and gamma_levels[0] == 0.0
            and len(ils_levels) == 3
            and len(gene_tree_levels) == 4
            and len(positive_pair_types) == 3
        )
        if supervisor_grid:
            offset = deterministic_seed(base_seed, f"selection-offset|{backbone_id}") % 12
            positive_slot = 0
            schedule: list[int] = []
            for slot in range(12):
                gamma = gamma_levels[(slot + offset) % 3]
                ils = ils_levels[((slot // 3) + slot + offset) % 3]
                gene_tree_count = gene_tree_levels[(slot + offset) % 4]
                if gamma == 0.0:
                    pair_type = "none"
                    replicate = ((slot // 3) + offset) % 3 + 1
                else:
                    pair_type = positive_pair_types[(positive_slot + offset) % 3]
                    positive_slot += 1
                    replicate = 1
                candidates = group.loc[
                    np.isclose(group["gamma"].astype(float), gamma)
                    & np.isclose(group["ils_length"].astype(float), ils)
                    & (group["gene_tree_count"].astype(int) == gene_tree_count)
                    & (group["pair_type"].astype(str) == pair_type)
                    & (group["replicate"].astype(int) == replicate)
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected one supervisor-grid candidate for {backbone_id}, found {len(candidates)}"
                    )
                schedule.append(int(candidates.index[0]))
            if runs_per_backbone == 12:
                selected_indices.extend(schedule)
            else:
                tree_size = int(group["tree_size"].iloc[0])
                deadline_schedules.setdefault(tree_size, []).append(
                    (str(backbone_id), schedule)
                )
            continue

        availability = {
            factor: group[factor].astype(str).value_counts().to_dict() for factor in factors
        }
        factor_counts = {
            factor: {level: 0 for level in levels} for factor, levels in availability.items()
        }
        used_scenarios: set[str] = set()
        remaining = set(group.index.tolist())
        for selection_index in range(runs_per_backbone):
            next_total = selection_index + 1

            def candidate_key(index: int) -> tuple[float, int]:
                row = group.loc[index]
                penalty = 0.0
                for factor in factors:
                    level = str(row[factor])
                    proposed = dict(factor_counts[factor])
                    proposed[level] += 1
                    total_available = float(sum(availability[factor].values()))
                    for candidate_level, count in proposed.items():
                        target = next_total * availability[factor][candidate_level] / total_available
                        penalty += ((count - target) ** 2) / max(target, 0.5)
                if str(row["scenario_group"]) in used_scenarios:
                    penalty += 1_000_000.0
                tie = deterministic_seed(
                    base_seed, f"selection|{backbone_id}|{row['simulation_id']}"
                )
                return penalty, tie

            chosen = min(remaining, key=candidate_key)
            remaining.remove(chosen)
            selected_indices.append(chosen)
            selected_row = group.loc[chosen]
            used_scenarios.add(str(selected_row["scenario_group"]))
            for factor in factors:
                factor_counts[factor][str(selected_row[factor])] += 1

    # A deadline backbone uses one of four consecutive three-cell blocks from
    # its compact 12-cell schedule. Every block contains all gamma and ILS
    # levels. Dynamic programming chooses blocks before simulation so that,
    # within each taxon size, gene-tree-count and positive-pair margins differ
    # by at most one whenever that balance is feasible.
    for tree_size, entries in sorted(deadline_schedules.items()):
        gene_tree_levels = sorted(manifest["gene_tree_count"].astype(int).unique())
        positive_pair_types = sorted(
            value
            for value in manifest["pair_type"].astype(str).unique()
            if value != "none"
        )
        dimensions = len(gene_tree_levels) + len(positive_pair_types)
        states: dict[tuple[int, ...], tuple[int, ...]] = {
            (0,) * dimensions: ()
        }
        for backbone_id, schedule in sorted(entries):
            profiles: list[tuple[int, ...]] = []
            for block in range(4):
                rows = manifest.loc[schedule[3 * block : 3 * block + 3]]
                profile = tuple(
                    int((rows["gene_tree_count"].astype(int) == level).sum())
                    for level in gene_tree_levels
                ) + tuple(
                    int((rows["pair_type"].astype(str) == pair_type).sum())
                    for pair_type in positive_pair_types
                )
                profiles.append(profile)
            next_states: dict[tuple[int, ...], tuple[int, ...]] = {}
            for state, path in states.items():
                for block, profile in enumerate(profiles):
                    next_state = tuple(a + b for a, b in zip(state, profile))
                    next_path = path + (block,)
                    previous = next_states.get(next_state)
                    if previous is None or next_path < previous:
                        next_states[next_state] = next_path
            states = next_states

        backbone_count = len(entries)
        gene_target = 3.0 * backbone_count / len(gene_tree_levels)
        pair_target = 2.0 * backbone_count / len(positive_pair_types)

        def deadline_score(item: tuple[tuple[int, ...], tuple[int, ...]]) -> tuple:
            state, path = item
            gene_counts = state[: len(gene_tree_levels)]
            pair_counts = state[len(gene_tree_levels) :]
            return (
                max(gene_counts) - min(gene_counts),
                max(pair_counts) - min(pair_counts),
                sum((value - gene_target) ** 2 for value in gene_counts)
                + sum((value - pair_target) ** 2 for value in pair_counts),
                path,
            )

        _, selected_blocks = min(states.items(), key=deadline_score)
        for (_, schedule), block in zip(sorted(entries), selected_blocks):
            selected_indices.extend(schedule[3 * block : 3 * block + 3])

    result = manifest.loc[selected_indices].copy()
    result["source_manifest_row"] = result.index.to_numpy(dtype=int) + 1
    result["analysis_selected"] = True
    result = result.sort_values("source_manifest_row").reset_index(drop=True)
    result["analysis_selection_rank"] = (
        result.groupby("backbone_id", sort=False).cumcount() + 1
    )
    return result


def make_manifest(config_path: str | Path) -> pd.DataFrame:
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    direction_strategy = str(cfg.get("direction_strategy", "configured"))
    if direction_strategy not in {"configured", "distance_stratified"}:
        raise ValueError("direction_strategy must be configured or distance_stratified")
    backbone_records = _build_backbone_records(cfg)
    rows: list[dict] = []
    scenario_index = 0

    if "scenario_design" in cfg:
        settings = [
            (float(item["ils_length"]), float(item["gamma"]), int(item["gene_tree_count"]))
            for item in cfg["scenario_design"]
        ]
    else:
        settings = [
            (float(ils), float(gamma), int(g))
            for ils, gamma, g in itertools.product(
                cfg["ils_levels"], cfg["gammas"], cfg["gene_tree_counts"]
            )
        ]

    assignment = str(cfg.get("backbone_assignment", "rotate"))
    if assignment not in {"rotate", "cross_product"}:
        raise ValueError("backbone_assignment must be rotate or cross_product")
    if assignment == "rotate" and direction_strategy == "distance_stratified":
        raise ValueError("distance_stratified directions require backbone_assignment: cross_product")

    backbone_cache: dict[tuple[str, float], tuple[BackboneNode, str]] = {}
    network_cache: dict[tuple[str, float, float, str, str], str] = {}

    def add_scenario(backbone_record: dict, ils: float, gamma: float, g: int, donor, recipient, pair_type):
        backbone_key = (backbone_record["backbone_id"], ils)
        if backbone_key not in backbone_cache:
            backbone = backbone_from_topology(
                backbone_record["topology"], ils, float(cfg.get("pendant_base", 1.0))
            )
            backbone_cache[backbone_key] = (backbone, render_backbone(backbone))
        backbone, backbone_newick = backbone_cache[backbone_key]
        direction_label = "none" if donor is None else f"{donor}>{recipient}"
        group_key = (
            f"{backbone_record['backbone_id']}|{ils:.8g}|{gamma:.8g}|{g}|{direction_label}|{pair_type}"
        )
        network_key = (
            backbone_record["backbone_id"],
            ils,
            gamma,
            str(donor or "none"),
            str(recipient or "none"),
        )
        if network_key not in network_cache:
            network_cache[network_key] = render_network(
                backbone, donor, recipient, gamma, float(cfg["gene_flow_height"])
            )
        network_newick = network_cache[network_key]
        for replicate in range(1, _replicates_for_gamma(cfg, gamma) + 1):
            pair_suffix = "" if pair_type in {"none", "configured"} else f"__{pair_type}"
            simulation_id = (
                f"{cfg['experiment_id']}__{backbone_record['backbone_id']}"
                f"__ils{ils:.2f}__g{gamma:.2f}__G{g}"
                f"__r{replicate}__{direction_label.replace('>', 'to')}{pair_suffix}"
            )
            rows.append(
                {
                    "simulation_id": simulation_id,
                    "experiment_id": cfg["experiment_id"],
                    "taxon_set_id": backbone_record["taxon_set_id"],
                    "tree_size": backbone_record["tree_size"],
                    "triplets_per_simulation": backbone_record["triplets_per_simulation"],
                    "backbone_id": backbone_record["backbone_id"],
                    "backbone_shape": backbone_record["backbone_shape"],
                    "backbone_shape_class": backbone_record["backbone_shape_class"],
                    "backbone_balance_score": backbone_record["backbone_balance_score"],
                    "backbone_balance_bin": backbone_record["backbone_balance_bin"],
                    "backbone_newick": backbone_newick,
                    "network_newick": network_newick,
                    "donor": donor or "none",
                    "recipient": recipient or "none",
                    "direction": direction_label,
                    "pair_type": pair_type,
                    "gamma": gamma,
                    "ils_length": ils,
                    "gene_tree_count": g,
                    "replicate": replicate,
                    "scenario_group": group_key,
                    "seed": deterministic_seed(int(cfg["base_seed"]), simulation_id),
                }
            )

    if assignment == "rotate":
        taxa = backbone_records[0]["taxa"]
        directions = parse_directions(cfg.get("directions", "all"), taxa)
        for ils, gamma, g in settings:
            if gamma == 0.0 and cfg.get("deduplicate_gamma_zero", True):
                scenario_directions = [(None, None)]
            else:
                scenario_directions = directions
            for donor, recipient in scenario_directions:
                backbone_record = backbone_records[scenario_index % len(backbone_records)]
                scenario_index += 1
                pair_type = "none" if donor is None else "configured"
                add_scenario(backbone_record, ils, gamma, g, donor, recipient, pair_type)
    else:
        for backbone_record in backbone_records:
            if direction_strategy == "distance_stratified":
                selected_directions = distance_stratified_directions(
                    backbone_record["topology"], int(cfg["base_seed"]), backbone_record["backbone_id"]
                )
            else:
                directions = parse_directions(
                    cfg.get("directions", "all"), backbone_record["taxa"]
                )
                selected_directions = [(donor, recipient, "configured") for donor, recipient in directions]
            for ils, gamma, g in settings:
                if gamma == 0.0 and cfg.get("deduplicate_gamma_zero", True):
                    scenario_directions = [(None, None, "none")]
                else:
                    scenario_directions = selected_directions
                for donor, recipient, pair_type in scenario_directions:
                    add_scenario(backbone_record, ils, gamma, g, donor, recipient, pair_type)
    return pd.DataFrame(rows)
