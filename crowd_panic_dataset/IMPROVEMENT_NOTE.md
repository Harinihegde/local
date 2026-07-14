# Post-baseline improvement note

The original 266-clip held-out test was evaluated once and is frozen. No
post-baseline decision in this note used its labels.

On five stratified folds of the original 1,064 training clips, the baseline
random forest using the initial summary features achieved macro-F1 0.7360.
An Extra Trees classifier using the complete temporal summary feature set
(including flow/coherence/speed peak and trend features) achieved macro-F1
0.7507. This candidate is retained as an experiment note, not the default
pipeline.

This is evidence of a likely improvement, not a replacement test result. The
default pipeline deliberately remains the Random Forest configuration that
generated the recorded 80.83% result. A new independent evaluation set is
required before promoting the candidate or claiming a new final accuracy.
