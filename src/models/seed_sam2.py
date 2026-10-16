"""Register a SAM 2.1 checkpoint in the model registry and promote it to production.

    python -m src.models.seed_sam2 --checkpoint weights/sam2.1_t.pt --version 2.1-tiny

Same lifecycle as the tabular model: artifact → storage → registry → promote.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import requests

from src.serving.dependency import get_artifact_storage

BASE_URL = os.environ.get("MSP_BASE_URL", "http://localhost:8000")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="weights/sam2.1_t.pt")
    ap.add_argument("--name", default="sam2-video")
    ap.add_argument("--version", default="2.1-tiny")
    args = ap.parse_args()

    blob = Path(args.checkpoint).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()[:12]
    uri = f"local://{args.name}/{args.version}/{digest}.pt"
    get_artifact_storage().save(uri, blob)
    print(f"stored {len(blob) / 2**20:.0f} MB at {uri}")

    requests.post(f"{BASE_URL}/models", json={"name": args.name, "description": "SAM 2.1 video object segmentation"})
    r = requests.post(
        f"{BASE_URL}/models/{args.name}/versions",
        json={"version": args.version, "framework": "sam2", "model_format": "pytorch-checkpoint", "artifact_uri": uri},
    )
    print("register:", r.status_code, r.text[:200])
    r = requests.post(f"{BASE_URL}/models/{args.name}/versions/{args.version}/promote", timeout=600)
    print("promote:", r.status_code, r.text[:200])
    print(
        f"\ntry it:\n  curl -F video=@clip.mp4 "
        f'-F \'prompts={{"objects":[{{"obj_id":1,"points":[[400,220]]}}]}}\' '
        f"{BASE_URL}/models/{args.name}/segment-video"
    )


if __name__ == "__main__":
    main()
