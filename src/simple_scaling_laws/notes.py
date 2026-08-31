"""Structured, machine-readable warnings carried through fitting and into the saved artifact.

The package always returns a fit when one is mathematically possible; anything it is unhappy about is recorded
here instead of raised, so an automated caller can inspect the reasons a prediction should be distrusted
rather than parsing log text.
"""

from __future__ import annotations

import dataclasses
from typing import Any

#: Severity levels, in increasing order of seriousness.
SEVERITIES: tuple[str, ...] = ("info", "warning", "error")


@dataclasses.dataclass(frozen=True, slots=True)
class Note:
    """One diagnostic note about a fit.

    Attributes:
        code: Stable machine-readable identifier, e.g. ``"too_few_configurations"``.
        severity: One of :data:`SEVERITIES`.
        message: Human-readable explanation.
        details: Any structured, JSON-serializable supporting values.

    Examples:
        >>> note = Note("too_few_configurations", "warning", "Only 2 configurations.", {"n": 2})
        >>> print(note)
        [warning] too_few_configurations: Only 2 configurations.
        >>> Note.from_dict(note.to_dict()) == note
        True
    """

    code: str
    severity: str
    message: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the severity level.

        Raises:
            ValueError: If ``severity`` is not one of :data:`SEVERITIES`.
        """
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def __str__(self) -> str:
        """Render as ``[severity] code: message``."""
        return f"[{self.severity}] {self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Note:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            code=data["code"],
            severity=data["severity"],
            message=data["message"],
            details=dict(data.get("details") or {}),
        )
