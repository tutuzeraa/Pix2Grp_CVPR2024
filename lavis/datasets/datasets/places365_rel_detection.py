import os
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Optional: load the canonical VG‑150 vocab via `load_categories_info` (same
# helper used by oiv6_rel_detection).  We fall back to a tiny hard‑coded list so
# inference never fails even if the JSON is missing.
# -----------------------------------------------------------------------------

try:
    from lavis.datasets.datasets.oiv6_rel_detection import load_categories_info  # noqa: E501
except (ImportError, ModuleNotFoundError):
    print("DEU PAU AQUI!!!\n\n\n")
    load_categories_info = None  # type: ignore

_FALLBACK_OBJS = [
    "__background__", "person", "car", "tree", "building", "road", "sky",
    "grass", "dog", "cat",
]
_FALLBACK_RELS = [
    "__no_relation__", "on", "under", "inside", "behind", "in front of",
    "above", "below", "next to", "over", "between",
]

__all__ = ["Places365EvalDataset"]

# -----------------------------------------------------------------------------
# Helper to gather image paths
# -----------------------------------------------------------------------------

def _discover_images(root: Path, image_list_file: Optional[Path] = None) -> List[Path]:
    """Return all image paths under *root* (optionally restricted by list)."""
    if image_list_file and image_list_file.exists():
        with image_list_file.open("r") as f:
            rel = [ln.strip() for ln in f if ln.strip()]
        paths = [root / p for p in rel]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No images found under {root} (list file = {image_list_file}).")
    return sorted(paths)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class Places365EvalDataset(Dataset):
    # existing docstring omitted for brevity(Dataset):
    """Minimal inference‑only dataset for Pix2Grp on **Places365** images.

    Accepts all extra kwargs passed by LAVIS' generic builders so signature
    mismatches don’t crash training/eval scripts.
    """

    def __init__(
        self,
        vis_processor,
        text_processor=None,
        # These two are supplied when the class is instantiated via
        # `BaseDatasetBuilder.build()` (generic builder). We honour them.
        vis_root: Optional[str] = None,
        ann_paths: Optional[list] = None,  # ignored – no annotations for Places365
        # When called from a *custom* builder we get a full dataset_cfg.
        dataset_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,  # absorb any other builder‑specific args
    ) -> None:
        super().__init__()

        print("DEBUG-INIT  kwargs:", kwargs)

        # ------------------------------------------------------------------
        # Build a unified config dict we can rely on later
        # ------------------------------------------------------------------
        if dataset_cfg is None:
            if vis_root is None:
                raise ValueError("Either dataset_cfg or vis_root must be provided.")
            dataset_cfg = {
                "build_info": {
                    "images": {"storage": "cache/rcpd"},
                    # "image_lists" :  {"storage":  "/home/artur.barros/Pix2Grp/Pix2Grp_CVPR2024/cache/places365/places8_2.csv"}
                }
            }
        self.config = dataset_cfg  # keep a reference

        build_info = dataset_cfg.get("build_info", {})

        # ------------------------------------------------------------------
        # Resolve image root
        # ------------------------------------------------------------------
        img_cfg = build_info.get("images", {})
        self.root = Path(img_cfg.get("storage", "")).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Image root {self.root} does not exist.")

        # ------------------------------------------------------------------
        # Optional list or CSV file to subset images for inference
        # ------------------------------------------------------------------
        print(f"build_info is:\n {build_info}\n")
        list_cfg  = build_info.get("image_lists", {})
        print(f"list config is {list_cfg}\n")
        list_path = list_cfg.get("storage")
        list_file = Path(list_path).expanduser().resolve() if list_path else None

        if list_file and list_file.exists():
            suffix = list_file.suffix.lower()

            if suffix == ".csv":
                print(f"DEBUG-INIT using CSV list: {list_file}")
                from csv import DictReader
                rel_paths = []
                with open(list_file, newline="") as cf:
                    reader = DictReader(cf)
                    for row in reader:
                        p = row.get("filepath") or row.get("image_id") or row.get("filename")
                        if p:
                            rel_paths.append(p)
                resolved = []
                for p in rel_paths:
                    p_path = Path(p)
                    if p_path.is_absolute() and p_path.exists():
                        resolved.append(p_path)
                    else:
                        # strip everything up to the last occurrence of "data_256/"
                        p_str = p
                        if "data_256/" in p_str:
                            rel = p_str.rsplit("data_256/", 1)[-1]
                        else:
                            rel = p_str
                        full = self.root / rel
                        if full.exists():
                            resolved.append(full)
                        else:
                            warnings.warn(f"CSV entry {p} -> {full} not found; skipping.")
                self.img_paths = resolved

            else:
                print(f"DEBUG-INIT using text list: {list_file}")
                self.img_paths = _discover_images(self.root, list_file)
        else:
            print("DEBUG-INIT no list provided, discovering all images")
            self.img_paths = _discover_images(self.root, None)
        # ------------------------------------------------------------------


        # ------------------------------------------------------------------
        # Vocabulary – VG‑150 categories to keep Pix2Grp’s post‑processing happy
        # ------------------------------------------------------------------
        cat_cfg = build_info.get("categories", {})
        cat_path = Path(cat_cfg.get("storage", "")).expanduser().resolve() if cat_cfg.get("storage") else None
        if cat_path and cat_path.exists() and load_categories_info is not None:
            try:
                objs, rels, *_ = load_categories_info(str(cat_path))
                self.object_names = objs
                self.predicate_names = rels
            except Exception as e:  # pragma: no cover
                warnings.warn(f"Failed to load categories from {cat_path}: {e}. Using fallback lists.")
                self.object_names = _FALLBACK_OBJS
                self.predicate_names = _FALLBACK_RELS
        else:
            self.object_names = _FALLBACK_OBJS
            self.predicate_names = _FALLBACK_RELS

        # Processors
        self.vis_processor = vis_processor
        self.text_processor = text_processor  # kept for API compatibility

        # print("DEBUG-INIT  self.collater id:", id(self.collater))

    # ------------------------------------------------------------------
    # PyTorch dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load {img_path}: {e}; using a blank image.")
            img = Image.new("RGB", (224, 224))

        w, h = img.size
        img_rel_path = str(img_path.relative_to(self.root))

        dummy_target: Dict[str, Any] = {
            "image_id":   img_rel_path,
            "size":       torch.tensor([h, w]),
            "orig_size":  torch.tensor([h, w]),
            "boxes":       torch.zeros((0, 4), dtype=torch.float32),
            "labels":      torch.zeros((0,),    dtype=torch.int64),
            "det_labels":  torch.zeros((0,),    dtype=torch.int64),  # << add this
            "rel_tripets": torch.zeros((0, 3), dtype=torch.int64),
            "det_labels_texts": [],
            "rel_labels_texts": []
        }

        # `blip_det_image_eval` expects (image, target) -> (Tensor, target)
        img_tensor, _ = self.vis_processor(img, dummy_target)

        sample = {
            "image": img_tensor,
            "target": dummy_target,           # singular – matches VG datasets
            "targets": dummy_target,
            "image_id": img_rel_path,
            "height": h,
            "width": w,
        }

        # print("DEBUG-GET   sample keys:", list(sample.keys()))

        return sample

    # ------------------------------------------------------------------
    # Collate helper so variable‑size images won’t break DDP gather
    # ------------------------------------------------------------------

    def collater(self, batch):
        batch_dict = {
            "image":     torch.stack([b["image"] for b in batch]),
            "targets":   [b["targets"] for b in batch],
            "image_id":  [b["image_id"] for b in batch],
        }
        # DEBUG — show once per run
        if not hasattr(self, "_collate_shown"):
            # print("DEBUG-COLLATE batch keys:", list(batch_dict.keys()))
            self._collate_shown = True
        return batch_dict

    collate_fn = collater

    # Expose VG vocab to model / evaluator
    @property
    def obj_classes(self):
        return self.object_names

    @property
    def rel_classes(self):
        return self.predicate_names
