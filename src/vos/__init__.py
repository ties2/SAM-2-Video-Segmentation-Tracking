"""Video Object Segmentation (VOS) core.

This package is the computer-vision half of the platform. It is deliberately
independent of FastAPI / SQLAlchemy so it can be used from a notebook, a CLI,
a batch job or the serving layer (`src/serving`).

Modules
-------
types       Sequence / Prompt / SegmentationResult data classes
data        DAVIS, SA-V and synthetic sequence loaders
prompts     click / box / mask prompts derived from a ground-truth mask
rle         COCO-compatible run-length encoding (no pycocotools dependency)
metrics     DAVIS-standard J (region IoU), F (boundary F-measure), J&F, recall, decay
stability   ground-truth-free temporal-consistency metrics (flicker, drop-outs, drift)
synthetic   stress-test videos with exact ground truth (occlusion, distractor, exit, fast motion)
predictor   SAM 2.1 video predictor wrapper with a persistent per-frame feature cache
benchmark   config-driven experiment runner that writes CSV / JSON results
visualize   overlays, alpha-matte export, MP4 / GIF writers
"""
