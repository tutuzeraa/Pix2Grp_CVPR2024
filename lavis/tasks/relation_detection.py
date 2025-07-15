"""
Streams PGSG graphs to JSON-Lines during evaluation.

Output root (hard-coded): /home/artur.barros/Pix2Grp/Pix2Grp_CVPR2024/zzz
Each GPU/rank writes its own files:
    zzz/json_graphs/graphs_rank-<R>_part-<K>.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import torch

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.detection import DetectionTask
from lavis.tasks.evaluation.sgg_oi_eval import OpenImageSGGEvaluator
import lavis.tasks.evaluation.comm as comm

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
_DUMP_ROOT   = Path("/hadatasets/artur.barros/preds_places8/agora_sim")
_FLUSH_EVERY = 200          # graphs per file   ← increase freely

logger = logging.getLogger(__name__)


@registry.register_task("relation_detection")
class RelationDetectionTask(DetectionTask):
    """Relation-detection task that *streams* graphs to disk without loss."""

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        report_metric=True,
        experiments_mode="sggen",
        generation_mode="sampling",
        cate_dict_url="",
        zeroshot_cfg=None,
    ):
        super().__init__(
            num_beams, max_len, min_len, evaluate, report_metric, cate_dict_url
        )

        self.num_beams            = num_beams
        self.max_len              = max_len
        self.min_len              = min_len
        self.use_nucleus_sampling = generation_mode == "sampling"
        self.experiments_mode     = experiments_mode

        # we keep metrics disabled – but evaluator is left here if needed
        self.evaluator = OpenImageSGGEvaluator(
            self.cate_dict,
            eval_post_proc=False,
            eval_types=["bbox", "relation"],
            zeroshot_cfg=zeroshot_cfg,
        )

        # per-rank buffers
        self._cached_graphs: list[dict] = []
        self._part_idx: int            = 406

    # ------------------------------------------------------------------ #
    # helper – flush
    # ------------------------------------------------------------------ #
    def _flush_graphs(self, *, force: bool = False):
        """Write buffer when ≥ _FLUSH_EVERY or *force* is True."""
        if not self._cached_graphs:
            return
        if not force and len(self._cached_graphs) < _FLUSH_EVERY:
            return

        out_root = _DUMP_ROOT / "json_graphs"
        out_root.mkdir(parents=True, exist_ok=True)

        out_file = out_root / (
            f"graphs_rank-{comm.get_rank():02d}_part-{self._part_idx:04d}.jsonl"
        )
        with out_file.open("w", encoding="utf8") as fh:
            for rec in self._cached_graphs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            "[rank %d] flushed %d graphs → %s",
            comm.get_rank(),
            len(self._cached_graphs),
            out_file,
        )
        self._cached_graphs.clear()
        self._part_idx += 1

    # ------------------------------------------------------------------ #
    # validation loop
    # ------------------------------------------------------------------ #
    def valid_step(self, model, samples):
        """Generate predictions and buffer per-image graphs."""
        if self.experiments_mode == "sggen":
            preds, gts, img_info, graphs = model.generate(
                samples,
                use_nucleus_sampling=self.use_nucleus_sampling,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                repetition_penalty=1.5,
                num_captions=1,
            )
        else:                               # sgcls
            preds, gts, img_info = model.sgg_cls(samples)
            graphs = []

        # ------------- buffer (all ranks, no filtering) ------------------
        for i, iid in enumerate(img_info):
            g = graphs[i] if i < len(graphs) else []        # guarantee one entry / image
            self._cached_graphs.append({"image_id": iid, "triplets": g})
        self._flush_graphs()                                # may flush now 

        # ------------- return dummy metrics payload ----------------------
        res: List[dict] = []
        for i, iid in enumerate(samples["image_id"]):
            res.append(
                {
                    "predictions":   preds[i],
                    "ground_truths": gts[i],
                    "image_id":      iid,
                    "image_info":    img_info[i],
                }
            )
        return res

    # ------------------------------------------------------------------ #
    # after evaluation – force remainder flush
    # ------------------------------------------------------------------ #
    def after_evaluation(self, *_args, **_kw):
        self._flush_graphs(force=True)   # every rank writes leftovers
        torch.distributed.barrier()      # sync workers
        return {"agg_metrics": 0.0}

    # ------------------------------------------------------------------ #
    # unused metric reporter
    # ------------------------------------------------------------------ #
    @main_process
    def _report_metrics(self, *_, **__):  # noqa: D401
        pass

    # ------------------------------------------------------------------ #
    # hydra hook
    # ------------------------------------------------------------------ #
    @classmethod
    def setup_task(cls, cfg):
        run = cfg.run_cfg
        return cls(
            num_beams       = run.num_beams,
            max_len         = run.max_len,
            min_len         = run.min_len,
            evaluate        = run.evaluate,
            report_metric   = run.get("report_metric", True),
            experiments_mode= run.get("experiments_mode", "sggen"),
            generation_mode = run.get("generation_mode", "sampling"),
            cate_dict_url   = run.get("cate_dict_url", ""),
            zeroshot_cfg    = run.get("zeroshot_cfg", None),
        )
