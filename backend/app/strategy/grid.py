import hashlib
import itertools
import json

from app.strategy.models import RangeSpec, Strategy

MAX_RANGES = 3
MAX_COMBOS = 200


def _find_ranges(node, path=()):
    if isinstance(node, dict):
        if set(node) == {"min", "max", "step"}:
            yield path
            return
        for k, v in node.items():
            yield from _find_ranges(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _find_ranges(v, (*path, i))


def _set(node, path, value):
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _get(node, path):
    for key in path:
        node = node[key]
    return node


def expand_grid(strategy: Strategy) -> list[Strategy]:
    data = strategy.model_dump()
    paths = list(_find_ranges(data))
    if not paths:
        return [strategy]
    if len(paths) > MAX_RANGES:
        raise ValueError(
            f"At most {MAX_RANGES} values can be ranges; this strategy has {len(paths)}"
        )
    specs = [RangeSpec(**_get(data, p)) for p in paths]
    total = _product(s.count() for s in specs)
    if total > MAX_COMBOS:
        raise ValueError(f"Tuning would test {total:,} combinations; the limit is {MAX_COMBOS}")
    axes = [s.values() for s in specs]
    out = []
    for combo in itertools.product(*axes):
        d = json.loads(json.dumps(data))
        for p, v in zip(paths, combo, strict=True):
            _set(d, p, v)
        out.append(Strategy.model_validate(d))
    return out


def _product(counts) -> int:
    total = 1
    for c in counts:
        total *= c
    return total


def combo_count(strategy: Strategy) -> int:
    """How many concrete strategies expand_grid would produce, without building them."""
    data = strategy.model_dump()
    return _product(RangeSpec(**_get(data, p)).count() for p in _find_ranges(data))


def _strip_numbers(node, key=None):
    if isinstance(node, dict):
        if set(node) == {"min", "max", "step"}:
            return "#"
        return {k: _strip_numbers(v, k) for k, v in node.items() if k != "name"}
    if isinstance(node, list):
        return [_strip_numbers(v) for v in node]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return "#"
    return node


def structure_key(strategy: Strategy) -> str:
    data = strategy.model_dump()
    for side in ("buy", "sell"):
        _drop_param_values(data[side])
    blob = json.dumps(_strip_numbers(data), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _drop_param_values(node):
    """Indicator params keep their names (structure) but lose their values."""
    if isinstance(node, dict):
        if node.get("kind") == "indicator":
            node["params"] = sorted(node.get("params", {}))
        for v in node.values():
            _drop_param_values(v)
    elif isinstance(node, list):
        for v in node:
            _drop_param_values(v)
