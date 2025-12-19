from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional


def _list_files(folder: Path, exts: Optional[Set[str]] = None) -> List[Path]:
    """List files in folder (non-recursive). Optionally filter by extensions (case-insensitive)."""
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    files = [p for p in folder.iterdir() if p.is_file()]
    if exts is None:
        return files

    exts_lower = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    return [p for p in files if p.suffix.lower() in exts_lower]


def _img_key_from_name(filename: str) -> str:
    """Key for imgs: use stem directly."""
    return Path(filename).stem


def _mask_key_from_name(filename: str) -> Optional[str]:
    """
    Key for masks: remove trailing '_mask' from stem.
    e.g., 'abc_mask.png' -> 'abc'
    If file doesn't end with '_mask', return None (ignored).
    """
    stem = Path(filename).stem
    if stem.endswith("_mask"):
        return stem[:-5]  # remove '_mask'
    return None


def sync_imgs_masks(
    root_dir: str | Path,
    imgs_dirname: str = "imgs",
    masks_dirname: str = "masks",
    allowed_exts: Optional[Set[str]] = None,
    dry_run: bool = True,
) -> Dict[str, object]:
    """
    Compare files in root_dir/imgs and root_dir/masks (mask files have '_mask' suffix in stem),
    then use the smaller side as the "primary" set, deleting extra files from the other side.

    Args:
        root_dir: dataset root containing imgs/ and masks/
        imgs_dirname: folder name for images
        masks_dirname: folder name for masks
        allowed_exts: optional set like {'.png', '.jpg'}; if None, use all extensions
        dry_run: if True, do not delete; just report actions

    Returns:
        A report dict with counts and the list of files that were/would be deleted.
    """
    root = Path(root_dir)
    imgs_dir = root / imgs_dirname
    masks_dir = root / masks_dirname

    imgs_files = _list_files(imgs_dir, allowed_exts)
    masks_files = _list_files(masks_dir, allowed_exts)

    # Build key maps
    imgs_map: Dict[str, Path] = {}
    for p in imgs_files:
        key = _img_key_from_name(p.name)
        imgs_map[key] = p  # if duplicate stems exist, last one wins

    masks_map: Dict[str, Path] = {}
    ignored_masks: List[Path] = []
    for p in masks_files:
        key = _mask_key_from_name(p.name)
        if key is None:
            ignored_masks.append(p)
            continue
        masks_map[key] = p

    imgs_keys = set(imgs_map.keys())
    masks_keys = set(masks_map.keys())

    # Decide primary side: smaller count wins (ties -> imgs as primary)
    if len(imgs_keys) <= len(masks_keys):
        primary = "imgs"
        primary_keys = imgs_keys
        other = "masks"
        other_map = masks_map
    else:
        primary = "masks"
        primary_keys = masks_keys
        other = "imgs"
        other_map = imgs_map

    # Anything in the other side not in primary_keys should be removed
    extra_keys = set(other_map.keys()) - set(primary_keys)
    to_delete = [other_map[k] for k in sorted(extra_keys)]

    deleted: List[str] = []
    for p in to_delete:
        if dry_run:
            deleted.append(str(p))
        else:
            p.unlink(missing_ok=True)
            deleted.append(str(p))

    report = {
        "root_dir": str(root.resolve()),
        "imgs_dir": str(imgs_dir.resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "allowed_exts": None if allowed_exts is None else sorted(list(allowed_exts)),
        "dry_run": dry_run,
        "primary_side": primary,
        "counts": {
            "imgs_files": len(imgs_files),
            "masks_files": len(masks_files),
            "imgs_keys": len(imgs_keys),
            "masks_keys": len(masks_keys),
            "ignored_masks_without__mask_suffix": len(ignored_masks),
            "to_delete": len(to_delete),
        },
        "deleted_or_would_delete": deleted,
        "note": (
            "Masks not ending with '_mask' were ignored and never deleted by this script."
            if ignored_masks
            else "No ignored mask files."
        ),
    }
    print(root_dir)
    print(report["primary_side"], report["counts"])
    print("\n".join(report["deleted_or_would_delete"][:20]))
    print("\n")
    return report
