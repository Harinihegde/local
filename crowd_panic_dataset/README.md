# Crowd Panic Dataset — detection to risk classification

This is isolated from the earlier UMN work. It first reads the actual MP4
headers, then creates one reproducible stratified 80/20 test split (seed
`20260714`). Test labels are never used for detector choice, feature tuning,
short-clip option selection, or model fitting.

The pipeline is detection (locally supplied YOLO COCO-person model) →
deterministic centroid tracking → people count, velocity, optical-flow and
motion-coherence features → train-only anomaly features → class-balanced
four-class random-forest risk classifier. It tests 16-frame, 32-frame, and
cross-clip normal-prior alternatives on an inner split of the training data;
the selected one alone is evaluated on the test set. Count drops are not
assumed: absolute crowd level, trends, and chaotic movement are all features.

Run the verified properties/split stage:

```bash
python3 run_pipeline.py --dataset /Users/harinihegde/Downloads/crowd_panic --output outputs --inventory-only
```

Then run the complete experiment with an already-available, manually
spot-checked person detector (the script never downloads one implicitly):

```bash
python3 run_pipeline.py --dataset /Users/harinihegde/Downloads/crowd_panic --output outputs --weights /absolute/path/to/person-model.pt
```

`outputs/results.json` provides the comparative candidates, final held-out
accuracy, and precision/recall/F1 for all classes including `Panic`.
