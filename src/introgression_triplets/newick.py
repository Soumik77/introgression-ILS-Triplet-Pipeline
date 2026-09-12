from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(eq=False)
class Node:
    """Minimal rooted Newick node.

    ``length`` is the length of the edge from this node to its parent.
    """

    name: str | None = None
    length: float | None = None
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list["Node"]:
        return [n for n in self.preorder() if n.is_leaf]

    def preorder(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.preorder()


class NewickError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.i = 0

    def parse(self) -> Node:
        node = self._subtree()
        self._ws()
        if self._peek() == ";":
            self.i += 1
        self._ws()
        if self.i != len(self.text):
            raise NewickError(f"Unexpected text at position {self.i}: {self.text[self.i:self.i+20]}")
        return node

    def _subtree(self) -> Node:
        self._ws()
        if self._peek() == "(":
            self.i += 1
            children = [self._subtree()]
            self._ws()
            while self._peek() == ",":
                self.i += 1
                children.append(self._subtree())
                self._ws()
            self._expect(")")
            name = self._label()
            length = self._length()
            node = Node(name=name or None, length=length, children=children)
            for child in children:
                child.parent = node
            return node
        name = self._label()
        if not name:
            raise NewickError(f"Expected leaf label at position {self.i}")
        return Node(name=name, length=self._length())

    def _label(self) -> str:
        self._ws()
        if self._peek() in "'\"":
            quote = self.text[self.i]
            self.i += 1
            start = self.i
            while self.i < len(self.text) and self.text[self.i] != quote:
                self.i += 1
            if self.i >= len(self.text):
                raise NewickError("Unterminated quoted label")
            value = self.text[start:self.i]
            self.i += 1
            return value
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ":,();":
            self.i += 1
        return self.text[start:self.i].strip()

    def _length(self) -> float | None:
        self._ws()
        if self._peek() != ":":
            return None
        self.i += 1
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ",();":
            self.i += 1
        raw = self.text[start:self.i].strip()
        if not raw:
            raise NewickError("Missing branch length")
        try:
            return float(raw)
        except ValueError as exc:
            raise NewickError(f"Invalid branch length: {raw}") from exc

    def _ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def _expect(self, token: str) -> None:
        self._ws()
        if self._peek() != token:
            raise NewickError(f"Expected {token!r} at position {self.i}")
        self.i += 1


def parse_newick(text: str) -> Node:
    return _Parser(text).parse()


def read_newick_lines(path: str) -> Iterator[Node]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield parse_newick(text)
            except NewickError as exc:
                raise NewickError(f"{path}:{line_number}: {exc}") from exc


def to_newick(node: Node, include_lengths: bool = False) -> str:
    if node.children:
        body = "(" + ",".join(to_newick(c, include_lengths).rstrip(";") for c in node.children) + ")"
        if node.name:
            body += node.name
    else:
        body = node.name or ""
    if include_lengths and node.length is not None:
        body += f":{node.length:.10g}"
    return body + (";" if node.parent is None else "")


def leaf_map(root: Node) -> dict[str, Node]:
    mapping: dict[str, Node] = {}
    for leaf in root.leaves():
        if leaf.name is None:
            raise NewickError("Unnamed leaf")
        if leaf.name in mapping:
            raise NewickError(f"Duplicate taxon label: {leaf.name}")
        mapping[leaf.name] = leaf
    return mapping


def ancestors(node: Node) -> list[Node]:
    result: list[Node] = []
    current: Node | None = node
    while current is not None:
        result.append(current)
        current = current.parent
    return result


def mrca(nodes: list[Node]) -> Node:
    if not nodes:
        raise ValueError("mrca requires at least one node")
    first_path = ancestors(nodes[0])
    other_sets = [set(ancestors(n)) for n in nodes[1:]]
    for candidate in first_path:
        if all(candidate in s for s in other_sets):
            return candidate
    raise NewickError("Tree is disconnected")


def edge_depth(node: Node) -> int:
    depth = 0
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def distance_to_ancestor(node: Node, ancestor: Node) -> float:
    distance = 0.0
    current = node
    while current is not ancestor:
        if current.parent is None:
            raise NewickError("Requested node is not an ancestor")
        if current.length is None:
            raise NewickError("Branch length is required for triplet features")
        distance += current.length
        current = current.parent
    return distance

