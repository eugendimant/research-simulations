"""Pre-committed evaluation set — sample and lock a fixed evaluation manifest.

The manifest fully determines the evaluation set: once written,
the same games are used for all backends, all optimization runs,
and all regression checks.

Usage:
    from socsim.bench.precommit import generate_precommit_manifest
    manifest = generate_precommit_manifest(seed=42, n=50)
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..games.families.money_request_family import MoneyRequestFamily, FamilySpec
from ..games.families.dedup import deduplicate_specs


def generate_precommit_manifest(
    seed: int = 42,
    n: int = 50,
    family: Optional[MoneyRequestFamily] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sample N games from a family using a fixed RNG seed and create a manifest.

    Parameters
    ----------
    seed : int
        RNG seed for reproducible sampling.
    n : int
        Number of games to include.
    family : MoneyRequestFamily, optional
        Game family to sample from. Defaults to MoneyRequestFamily().
    output_path : Path, optional
        If given, write manifest JSON here.

    Returns
    -------
    dict
        The complete manifest with hashes and full specs.
    """
    if family is None:
        family = MoneyRequestFamily()

    rng = np.random.default_rng(seed)
    specs = family.sample(rng, n)
    specs = deduplicate_specs(specs)

    manifest = family.generate_manifest(specs, seed)

    # Add a manifest-level hash for integrity checking
    manifest_str = json.dumps(manifest, sort_keys=True, ensure_ascii=True)
    manifest["manifest_hash"] = hashlib.sha256(manifest_str.encode()).hexdigest()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    return manifest


def load_precommit_manifest(path: Path, verify_integrity: bool = True) -> Dict[str, Any]:
    """Load and validate a precommit manifest.

    Parameters
    ----------
    path : Path
        Path to manifest JSON file.
    verify_integrity : bool
        If True, verify the manifest hash for tampering detection.

    Returns
    -------
    dict
        Validated manifest data.

    Raises
    ------
    ValueError
        If manifest is invalid or integrity check fails.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Required keys validation
    if "specs" not in data:
        raise ValueError("Invalid manifest: missing 'specs' key")
    if not isinstance(data["specs"], list):
        raise ValueError("Invalid manifest: 'specs' must be a list")
    if len(data["specs"]) == 0:
        raise ValueError("Invalid manifest: 'specs' is empty")

    # Validate each spec has required fields
    for i, spec in enumerate(data["specs"]):
        if "game_name" not in spec:
            raise ValueError(f"Invalid manifest: spec[{i}] missing 'game_name'")
        if "params" not in spec:
            raise ValueError(f"Invalid manifest: spec[{i}] missing 'params'")

    # Integrity verification
    if verify_integrity and "manifest_hash" in data:
        stored_hash = data.pop("manifest_hash")
        recomputed_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
        recomputed_hash = hashlib.sha256(recomputed_str.encode()).hexdigest()
        data["manifest_hash"] = stored_hash  # restore for caller
        if recomputed_hash != stored_hash:
            raise ValueError(
                f"Manifest integrity check failed: expected {stored_hash[:16]}..., "
                f"got {recomputed_hash[:16]}... — file may have been modified."
            )

    return data


def manifest_to_specs(manifest: Dict[str, Any]) -> List[FamilySpec]:
    """Convert a manifest back to FamilySpec objects."""
    return [
        FamilySpec(
            game_name=s["game_name"],
            params=s["params"],
            family_id=s.get("family_id", ""),
            spec_hash=s.get("spec_hash", ""),
        )
        for s in manifest["specs"]
    ]
