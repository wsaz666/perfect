"""Module dependency graph used to expand retrieval results."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .models import HdlModule


class ModuleGraph:
    def __init__(self, modules: Iterable[HdlModule]):
        self.modules = {module.name: module for module in modules}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)
        for module in self.modules.values():
            for instance in module.instances:
                self.outgoing[module.name].add(instance.module_name)
                self.incoming[instance.module_name].add(module.name)

    def neighbors(self, module_name: str) -> set[str]:
        return self.outgoing[module_name] | self.incoming[module_name]

    def expand(self, seeds: Iterable[str], depth: int = 1) -> dict[str, int]:
        """Return reachable modules mapped to their shortest graph distance."""
        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for seed in seeds:
            if seed:
                distances[seed] = 0
                queue.append((seed, 0))

        while queue:
            current, distance = queue.popleft()
            if distance >= depth:
                continue
            for neighbor in self.neighbors(current):
                next_distance = distance + 1
                if neighbor not in distances or next_distance < distances[neighbor]:
                    distances[neighbor] = next_distance
                    queue.append((neighbor, next_distance))
        return distances

    def dependency_lines(self) -> list[str]:
        lines = []
        for parent in sorted(self.outgoing):
            for child in sorted(self.outgoing[parent]):
                lines.append(f"{parent} -> {child}")
        return lines
