"""
Resolved stage configs: a base YAML plus dotted overrides, saved as the single file a stage reads.
"""
from pathlib import Path
from typing import Any, Collection, Mapping

from omegaconf import OmegaConf

_ABSENT = object()


def write_resolved_config(base: str | Path, overrides: Mapping[str, Any], dest: str | Path,
                          allow_new: Collection[str] = ()) -> Path:
    """
    Load `base`, set each dotted key of `overrides`, and save the result to `dest` (folders are created).

    Every override key must exist in `base`, except the keys listed in `allow_new`, which are added.
    Raises ValueError (before anything is written) for any other key.
    """
    config = OmegaConf.load(base)
    unknown = [key for key in overrides
               if key not in allow_new and OmegaConf.select(config, key, default=_ABSENT) is _ABSENT]
    if unknown:
        raise ValueError(f"override keys not in the base config {base}: {unknown}")
    OmegaConf.set_struct(config, False)
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, force_add=True)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, dest)
    return dest
