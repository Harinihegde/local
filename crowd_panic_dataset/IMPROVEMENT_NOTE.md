# Post-baseline improvement note

The original 266-clip held-out test was evaluated once and is frozen. No
post-baseline decision in this note used its labels.

On five stratified folds of the original 1,064 training clips, the baseline
random forest using the initial summary features achieved macro-F1 0.7360.
An Extra Trees classifier using the complete temporal summary feature set
(including flow/coherence/speed peak and trend features) achieved macro-F1
0.7507. The pipeline now uses that single train-only-selected change.

This is evidence of a likely improvement, not a replacement test result. A
new independent evaluation set is required before claiming a new final
accuracy; the prior test set has already been observed.
