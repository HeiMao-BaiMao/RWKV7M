"""Small HLO audit helpers for Phase 3 model-parallel verification."""

from __future__ import annotations

from dataclasses import dataclass
import re


_COLLECTIVE_NAMES = (
    "all-reduce",
    "all-gather",
    "all-to-all",
    "collective-permute",
    "reduce-scatter",
)


@dataclass(frozen=True)
class CollectiveAudit:
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, object]:
        return {"counts": dict(self.counts), "total": self.total}


def compiler_ir_text(lowered, *, dialect="hlo") -> str:
    """Return textual compiler IR without depending on one JAX IR wrapper."""
    if not hasattr(lowered, "compiler_ir"):
        as_text = getattr(lowered, "as_text", None)
        if as_text is None:
            raise TypeError("object exposes neither compiler_ir() nor as_text()")
        return as_text()
    ir = lowered.compiler_ir(dialect=dialect)
    as_hlo_text = getattr(ir, "as_hlo_text", None)
    return as_hlo_text() if as_hlo_text is not None else str(ir)


def audit_collectives(hlo_text: str) -> CollectiveAudit:
    """Count collective instructions in HLO text, one operation per line."""
    counts = {name: 0 for name in _COLLECTIVE_NAMES}
    for line in hlo_text.splitlines():
        normalized = line.lower().replace("_", "-")
        for name in _COLLECTIVE_NAMES:
            if re.search(rf"\b{re.escape(name)}\s*\(", normalized):
                counts[name] += 1
    return CollectiveAudit(counts=counts)


def audit_lowered_collectives(lowered, *, dialect="hlo") -> CollectiveAudit:
    return audit_collectives(compiler_ir_text(lowered, dialect=dialect))


__all__ = [
    "CollectiveAudit",
    "audit_collectives",
    "audit_lowered_collectives",
    "compiler_ir_text",
]
