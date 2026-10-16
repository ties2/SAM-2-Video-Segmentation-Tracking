# Upwork portfolio entry: copy & paste

**Project title** (69 / 70 chars)
```
AI Video Object Tracking & Segmentation with SAM 2 (PyTorch, FastAPI)
```

**Your role** (90 / 100 chars)
```
Computer Vision / ML Engineer: SAM 2 pipeline, J&F evaluation, FastAPI serving, monitoring
```

**Project description** (522 / 600 chars)
```
Goal: prove whether AI can keep a pixel-accurate mask on objects through a whole video (rotoscoping, privacy blur, tracking). Solution: SAM 2.1 pipeline with click/box prompts, DAVIS-standard J&F evaluation, stress tests (occlusion, exit/re-entry, look-alikes, fast motion) and label-free quality alerts, served via a FastAPI model registry with Prometheus/Grafana. Impact: J&F 0.987 from one click, instant recovery after a 12-frame occlusion, tracking failures flagged without labels (AUROC 0.97). 122 tests, Docker, CI.
```

**Skills and deliverables** (5)
`Computer Vision` · `Object Tracking` · `Image Segmentation` · `PyTorch` · `Deep Learning`

**Content blocks** (in this order; the first image becomes the thumbnail)
1. Image: `01_cover.jpg`
2. Video: `02_demo.mp4` (32 s)
3. Image: `03_results.png`
4. Image: `04_architecture.png`
5. Text:
   ```
   What's inside:
   • AI rotoscoping CLI: clicks/box in → alpha-matte PNGs + overlay MP4
   • Video segmentation REST API on a model registry (versioning, promotion, rollback)
   • DAVIS J&F / decay evaluation + synthetic stress tests with exact ground truth
   • Label-free quality alerts (object score, motion-compensated stability) → Prometheus/Grafana
   • Honest failure analysis: object leaving the frame → re-detected, lost, recovered
   ```
6. Link: your GitHub repository URL
7. File: `05_code_review.pdf`
