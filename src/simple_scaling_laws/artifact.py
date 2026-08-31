"""Reading and writing the ``.slaw`` artifact directory.

An artifact is a plain directory of four files, deliberately boring so it can be inspected, diffed
and archived without this package::

    experiment.slaw/
        manifest.json      what was fit, from what, over what range
        fits.json          point estimates, variance components, fit quality
        draws.npz          the uncertainty draws themselves, one array per parameter
        diagnostics.json   correlations, comparisons and every warning raised

Everything needed for prediction is inside; the original dataframe is not required.
"""

from __future__ import annotations

import json
import math
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

#: Version of the on-disk artifact layout. Bumped only for incompatible changes.
FORMAT_VERSION = "1"

#: File names inside an artifact directory.
MANIFEST_FILE = "manifest.json"
FITS_FILE = "fits.json"
DRAWS_FILE = "draws.npz"
DIAGNOSTICS_FILE = "diagnostics.json"

#: Conventional suffix for an artifact directory.
SUFFIX = ".slaw"


class ArtifactError(ValueError):
    """Raised when an artifact is missing, malformed, or of an unsupported format version."""


def package_version() -> str:
    """The installed version of this package, or ``"unknown"`` outside an installed environment."""
    try:
        return metadata.version("simple-scaling-laws")
    except metadata.PackageNotFoundError:  # pragma: no cover - only when running from a source tree
        return "unknown"


def jsonable(value: Any) -> Any:
    """Convert numpy scalars, arrays and non-finite floats into strict-JSON-safe values.

    ``NaN`` and infinities are written by Python's ``json`` module as bare ``NaN``/``Infinity``
    tokens, which are *not* valid JSON and are rejected by strict parsers in other languages. They
    become ``null`` here so the artifact stays portable.

    Args:
        value: Any nested structure of dicts, sequences, numpy values and scalars.

    Returns:
        The same structure with JSON-safe leaves.

    Examples:
        >>> jsonable({"a": np.float64(1.5), "b": [np.int64(2), float("nan")], "c": np.array([1.0])})
        {'a': 1.5, 'b': [2, None], 'c': [1.0]}
    """
    match value:
        case dict():
            return {str(k): jsonable(v) for k, v in value.items()}
        case str() | bytes():
            return value if isinstance(value, str) else value.decode()
        case np.ndarray():
            return [jsonable(v) for v in value.tolist()]
        case list() | tuple():
            return [jsonable(v) for v in value]
        case np.integer():
            return int(value)
        case np.floating() | float():
            return float(value) if math.isfinite(float(value)) else None
        case np.bool_():
            return bool(value)
        case _:
            return value


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as strict, human-readable JSON."""
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    """Read a JSON file, raising :class:`ArtifactError` when it is missing."""
    if not path.is_file():
        raise ArtifactError(f"Artifact is missing {path.name!r} (looked in {path.parent})")
    return json.loads(path.read_text())


def write_artifact(
    path: str | Path,
    manifest: dict[str, Any],
    fits: dict[str, Any],
    draws: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
) -> Path:
    """Write a complete artifact directory, creating or overwriting its four files.

    An existing artifact directory is overwritten in place. A non-empty directory that is *not* an
    artifact is refused, so a mistyped output path cannot scatter files into someone's home or
    source directory.

    Args:
        path: Destination directory, conventionally ending in ``.slaw``.
        manifest: Manifest contents.
        fits: Per-target fit contents.
        draws: Uncertainty draws, keyed ``"<target>/<parameter>"``.
        diagnostics: Diagnostics contents.

    Returns:
        The resolved artifact path.

    Raises:
        ArtifactError: If ``path`` is a non-empty directory that is not already an artifact.
    """
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not (path / MANIFEST_FILE).is_file():
        raise ArtifactError(
            f"{path} already exists, is not empty, and is not an artifact directory (it has no "
            f"{MANIFEST_FILE}). Refusing to write into it; choose a different output path."
        )
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / MANIFEST_FILE, manifest)
    write_json(path / FITS_FILE, fits)
    write_json(path / DIAGNOSTICS_FILE, diagnostics)
    np.savez_compressed(path / DRAWS_FILE, **draws)
    return path


def read_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Read an artifact directory.

    Args:
        path: The artifact directory.

    Returns:
        A ``(manifest, fits, draws, diagnostics)`` tuple.

    Raises:
        ArtifactError: If the directory is missing, incomplete, or written by an incompatible
            future version of the format.

    Examples:
        >>> import tempfile
        >>> directory = Path(tempfile.mkdtemp()) / "example.slaw"
        >>> _ = write_artifact(
        ...     directory,
        ...     manifest={"format_version": FORMAT_VERSION},
        ...     fits={"test_loss__ce": {"params": {"E": 1.0}}},
        ...     draws={"test_loss__ce/E": np.array([1.0, 1.1])},
        ...     diagnostics={"warnings": []},
        ... )
        >>> manifest, fits, draws, diagnostics = read_artifact(directory)
        >>> fits["test_loss__ce"]["params"]["E"], draws["test_loss__ce/E"].tolist()
        (1.0, [1.0, 1.1])
        >>> read_artifact(directory.parent / "missing.slaw")
        Traceback (most recent call last):
        ...
        simple_scaling_laws.artifact.ArtifactError: No artifact directory at ...missing.slaw
    """
    path = Path(path)
    if not path.is_dir():
        raise ArtifactError(f"No artifact directory at {path}")
    manifest = read_json(path / MANIFEST_FILE)
    version = str(manifest.get("format_version", ""))
    if version != FORMAT_VERSION:
        raise ArtifactError(
            f"Artifact at {path} uses format version {version!r}, but this package reads "
            f"version {FORMAT_VERSION!r}."
        )
    fits = read_json(path / FITS_FILE)
    diagnostics = read_json(path / DIAGNOSTICS_FILE)
    draws_path = path / DRAWS_FILE
    if not draws_path.is_file():
        raise ArtifactError(f"Artifact is missing {DRAWS_FILE!r} (looked in {path})")
    with np.load(draws_path) as handle:
        draws = {key: np.asarray(handle[key], dtype=float) for key in handle.files}
    return manifest, fits, draws, diagnostics
