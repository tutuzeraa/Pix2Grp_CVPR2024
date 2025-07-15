from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

__all__ = [
    "Triplet", "DatasetVocab", "DumpViewerConfig", "process_dump", "cli",
]

# ---------------------------------------------------------------------------
# Configuration / Constants
# ---------------------------------------------------------------------------
_LOG = logging.getLogger("dump_viewer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SAFE = re.compile(r"[^A-Za-z0-9]+")
PALETTE = (
    "red", "green", "blue", "yellow", "magenta", "cyan", "orange", "springgreen"
)

# ----------------------------------------------------------------------------
# VG150 VOCABULARY
# ----------------------------------------------------------------------------
VG150_OBJ: tuple[str] = (
    "__background__",  # índice 0
    "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear",
    "bed", "bench", "bicycle", "bike", "bird", "board", "boat", "book", "boot",
    "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet",
    "camera", "cap", "car", "carpet", "cat", "chair", "clock", "coat",
    "computer", "cow", "cup", "curtain", "desk", "dog", "door", "drawer",
    "dress", "ear", "elephant", "eye", "face", "fence", "finger",
    "fire hydrant", "fireplace", "flag", "floor", "flower", "food", "fork",
    "frisbee", "giraffe", "girl", "glove", "grass", "guitar", "hair", "hand",
    "hat", "head", "helmet", "horse", "house", "jacket", "jeans", "keyboard",
    "kite", "lamp", "laptop", "leaf", "leg", "light", "logo", "man",
    "microphone", "microwave", "mirror", "monitor", "mouse", "mouth", "neck",
    "nose", "orange", "oven", "paddle", "pants", "paper", "pen", "pencil",
    "person", "phone", "pillow", "pizza", "plant", "plate", "player", "pole",
    "pot", "racket", "remote", "road", "rock", "roof", "room", "rug", "sand",
    "screen", "seat", "sheep", "shelf", "shirt", "shoe", "shorts", "sidewalk",
    "sign", "sink", "skateboard", "ski", "snow", "sofa", "spoon", "sport ball",
    "stairs", "stick", "street", "surfboard", "table", "tail", "tie", "tile",
    "toilet", "toothbrush", "towel", "tower", "train", "tree", "truck",
    "trunk", "tv", "umbrella", "vase", "vegetable", "wall", "watch", "wheel",
    "window", "wine glass", "woman", "zebra",
) 

VG150_REL: tuple[str] = (
    "on", "wearing", "holding", "of", "in", "near", "behind", "under",
    "above", "across", "against", "along", "at", "next to", "by", "with",
    "beside", "beneath", "between", "inside of", "surrounding", "covering",
    "attached to", "overlapping", "part of", "made of", "using", "carrying",
    "eating", "drinking", "looking at", "facing", "crossing", "sitting on",
    "standing on", "lying on", "parked on", "growing on", "hanging from",
    "leaning on", "driving", "riding", "walking on", "running on",
    "flying over", "playing", "watching", "talking to", "coming out of",
)  

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Triplet:
    subject: str
    predicate: str
    object: str
    score: float

    def to_json(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "score": self.score,
        }

@dataclass
class DatasetVocab:
    objects: Sequence[str] = VG150_OBJ
    relations: Sequence[str] = VG150_REL

    def is_novel_obj(self, label: str) -> bool:
        return label not in self.objects

    def is_novel_rel(self, label: str) -> bool:
        return label not in self.relations

@dataclass
class DumpViewerConfig:
    input_dir: Path
    output_dir: Path
    score_thresh: float = 0.5
    reduce_mode: str = "max"  # or "mean"
    vocab: DatasetVocab = field(default_factory=DatasetVocab)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

NumberLike = Union[int, float]
TensorLike = Union[NumberLike, Sequence[NumberLike], torch.Tensor]


def slug(txt: str) -> str:
    """Simplifica string para uso em nome de arquivo."""
    return SAFE.sub("_", txt.lower())[:40] or "unk"


def reduce_scores(values: TensorLike, *, mode: str = "max") -> float:
    """Reduz Tensor/lista a um único escalar (max ou mean)."""
    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return 0.0
        fn = values.max if mode == "max" else values.mean
        return float(fn().item())

    if isinstance(values, (list, tuple)):
        if not values:
            return 0.0
        return float(max(values) if mode == "max" else sum(values) / len(values))

    return float(values)  


def tensor_to_list(x: TensorLike) -> list[float]:
    """Converte tensor ou número para lista python (para coordenadas)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (int, float)):
        return [float(x)]
    return list(x)


def xywh_to_xyxy(box: Sequence[float], width: int, height: int) -> list[float]:
    """Converte (cx, cy, w, h) para (x1, y1, x2, y2).
    Detecta automaticamente se as coordenadas estão normalizadas (0‑1).
    """
    cx, cy, bw, bh = box  # noqa: invalid‑name
    if max(cx, cy, bw, bh) <= 1.0:
        cx, bw = cx * width, bw * width
        cy, bh = cy * height, bh * height
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    return [max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)]


def draw_boxes(img: Image.Image, boxes: Iterable[Sequence[float]], labels: Iterable[str]):
    draw = ImageDraw.Draw(img)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = PALETTE[i % len(PALETTE)]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 2, box[1] + 2), label, fill=color)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

# def process_dump(pkl_file: Path, cfg: DumpViewerConfig):
#     """Processa um único arquivo .pkl e gera saídas."""
#     with pkl_file.open("rb") as fh:
#         dump = pickle.load(fh, encoding="latin1")

#     img_path: str | None = (dump.get("image_path") or [None])[0]
#     image_id: str = Path(img_path).stem if img_path else pkl_file.stem

#     # prepara diretórios de saída ------------------------------------------------
#     out_dir = cfg.output_dir / image_id
#     rel_dir = out_dir / "relations"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     rel_dir.mkdir(exist_ok=True)

#     # carrega imagem para desenho ------------------------------------------------
#     img, W, H = None, None, None
#     if img_path and Path(img_path).is_file():
#         img = Image.open(img_path).convert("RGB")
#         W, H = img.size

#     # percorre instâncias --------------------------------------------------------
#     triplets: list[Triplet] = []
#     all_boxes, all_labels = [], []
#     novel_objs, novel_rels = set(), set()

#     for inst in dump["batch_object_list"][0]:
#         sub = inst["sub"]["label"]
#         obj = inst["obj"]["label"]
#         pred = inst["predicate"]["label"]
#         score = reduce_scores(inst["predicate"]["pred_scores"], mode=cfg.reduce_mode)

#         if score < cfg.score_thresh:
#             continue  # ignora low‑score

#         triplets.append(Triplet(sub, pred, obj, score))

#         if W and (bbox := inst["sub"].get("boxes")) is not None:
#             sub_box = xywh_to_xyxy(tensor_to_list(bbox), W, H)
#             all_boxes.append(sub_box)
#             all_labels.append(sub)
#         if W and (bbox := inst["obj"].get("boxes")) is not None:
#             obj_box = xywh_to_xyxy(tensor_to_list(bbox), W, H)
#             all_boxes.append(obj_box)
#             all_labels.append(obj)

#         if img and all_boxes[-2:] and len(all_boxes) >= 2:
#             tmp = img.copy()
#             draw_boxes(tmp, all_boxes[-2:], all_labels[-2:])
#             tmp.save(rel_dir / f"{slug(sub)}_{slug(pred)}_{slug(obj)}.jpg")

#         if cfg.vocab.is_novel_obj(sub):
#             novel_objs.add(sub)
#         if cfg.vocab.is_novel_obj(obj):
#             novel_objs.add(obj)
#         if cfg.vocab.is_novel_rel(pred):
#             novel_rels.add(pred)

#     # persistência ----------------------------------------------------------------
#     (out_dir / "triplets.json").write_text(
#         json.dumps({"triplets": [t.to_json() for t in triplets]}, indent=2),
#         encoding="utf8",
#     )

#     if img and all_boxes:
#         draw_boxes(img, all_boxes, all_labels)
#         img.save(out_dir / "bbox.jpg")

#     _LOG.info(
#         "[%s] novel objs=%d novel rels=%d", image_id, len(novel_objs), len(novel_rels)
#     )

def process_dump(pkl_file: Path, cfg: DumpViewerConfig):
    """Processa um único arquivo .pkl com N imagens de uma inferência em batch."""
    with pkl_file.open("rb") as fh:
        dump = pickle.load(fh, encoding="latin1")

    # dump["batch_object_list"]: list of length B, each is a list of inst dicts
    # dump["image_path"]:      list of length B, each is the path to that image
    B = len(dump["batch_object_list"])
    image_paths = dump.get("image_path", [None]*B)
    predictions = dump.get("predictions", [None]*B)
    # ground_truths unused in inference-only

    for bi in range(B):
        img_path = Path(image_paths[bi]) if image_paths[bi] else None
        image_id = img_path.stem if img_path else f"batch_{bi}"

        # setup folders
        out_dir = cfg.output_dir / image_id
        rel_dir = out_dir / "relations"
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_dir.mkdir(exist_ok=True)

        # copy the original image
        if img_path and img_path.is_file():
            (out_dir / "image.jpg").write_bytes(img_path.read_bytes())

        # load for drawing
        img, W, H = None, None, None
        if img_path and img_path.is_file():
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

        # collect triplets and draw crops
        triplets: List[Triplet] = []
        all_boxes, all_labels = [], []

        for inst in dump["batch_object_list"][bi]:
            sub  = inst["sub"]["label"]
            obj  = inst["obj"]["label"]
            pred = inst["predicate"]["label"]
            score = reduce_scores(inst["predicate"]["pred_scores"], mode=cfg.reduce_mode)
            if score < cfg.score_thresh:
                continue

            triplets.append(Triplet(sub, pred, obj, score))

            # subject box
            if img and (bbox := inst["sub"].get("boxes")) is not None:
                sub_box = xywh_to_xyxy(tensor_to_list(bbox), W, H)
                tmp = img.copy()
                draw_boxes(tmp, [sub_box], [sub])
                tmp.save(rel_dir / f"{slug(sub)}_sub.jpg")
                all_boxes.append(sub_box); all_labels.append(sub)

            # object box
            if img and (bbox := inst["obj"].get("boxes")) is not None:
                obj_box = xywh_to_xyxy(tensor_to_list(bbox), W, H)
                tmp = img.copy()
                draw_boxes(tmp, [obj_box], [obj])
                tmp.save(rel_dir / f"{slug(obj)}_obj.jpg")
                all_boxes.append(obj_box); all_labels.append(obj)

            # combined crop
            if img and len(all_boxes) >= 2:
                tmp = img.copy()
                draw_boxes(tmp, all_boxes[-2:], all_labels[-2:])
                tmp.save(rel_dir / f"{slug(sub)}_{slug(pred)}_{slug(obj)}.jpg")

        # write triplets.json
        (out_dir / "triplets.json").write_text(
            json.dumps({"triplets": [t.to_json() for t in triplets]}, indent=2),
            encoding="utf8",
        )

        # full‐image box overlay
        if img and all_boxes:
            draw_boxes(img, all_boxes, all_labels)
            img.save(out_dir / "bbox.jpg")

        _LOG.info(
            "[%s] saved %d triplets",
            image_id, len(triplets),
        )


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def cli(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Visualiza dumps PGSG gerados pelo BLIP")
    parser.add_argument("--dump", type=Path, required=True, help="Pasta com *.pkl gerados pela inferência")
    parser.add_argument("--out", type=Path, required=True, help="Diretório de saída")
    parser.add_argument("--thresh", type=float, default=0.3, help="Score mínimo para considerar a tripla")
    parser.add_argument("--reduce", choices=["max", "mean"], default="max", help="Como reduzir scores multi‑token a escalar")
    args = parser.parse_args(argv)

    cfg = DumpViewerConfig(
        input_dir=args.dump,
        output_dir=args.out,
        score_thresh=args.thresh,
        reduce_mode=args.reduce,
    )

    pkls = sorted(cfg.input_dir.glob("vl_det_dump_*.pkl"))
    if not pkls:
        _LOG.warning("Nenhum arquivo *.pkl encontrado em %s", cfg.input_dir)
        return

    for pkl_file in tqdm(pkls, desc="Processando"):
        try:
            process_dump(pkl_file, cfg)
        except Exception as exc:  # pylint: disable=broad-except
            _LOG.error("%s → %s", pkl_file.name, exc, exc_info=False)

if __name__ == "__main__":
    cli()
