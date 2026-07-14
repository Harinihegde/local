# Crowd Panic Dataset handoff

## Reproducible design

The experiment uses a fixed 80/20 stratified clip split created with seed
`20260714`: 1,064 training clips and 266 held-out test clips. The test subset
is only passed to the final fitted classifier once. The MP4 inventory is made
before feature extraction and is saved in `outputs/video_properties_by_class.csv`.

## Short-clip adaptation

The UMN 200-frame causal window is deliberately not reused: this dataset's
observed clips can be as short as 60 frames. The script compares a 16-frame
local proxy, a 32-frame local proxy, and a cross-clip normal prior. It selects
one by inner-validation macro-F1 using only training clips, records all three
candidate scores, and evaluates the selected option on the test subset.

## Results

No classification metric is pre-filled in this document. Run the full command
in `README.md`, then cite `outputs/results.json` as the single source of the
held-out accuracy and per-class precision, recall, and F1. In particular,
report the `Panic` row alongside accuracy and do not claim the 90% target
unless that held-out output reaches it.

## What to inspect before reporting

Manually spot-check the detector's boxes on frames sampled across every class.
If the local model is not satisfactory, document the alternative and rerun the
entire feature stage; never compare detector alternatives using test labels.
