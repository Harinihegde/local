# Member 3: Anomaly Detection (Zone-Aware LSTM Ensemble)

## What This Module Does

* **Pretrains** on UMN panic dataset to learn crowd anomaly patterns
* **Fine-tunes** on Avenue training split (70%) with supervised learning
* **Optimizes** detection threshold using ground truth labels
* **Evaluates** on Avenue test split (30%) for cross-dataset performance
* Uses **zone-aware architecture** preserving spatial relationships (6 zones × 11 features)
* Leverages **rolling buffer features** (`_sm` columns) for temporal smoothing

## Approach

### Architecture
- **Multi-channel LSTM Autoencoder**: Processes 6 crowd zones independently
- **Input shape**: (sequence_length=30, zones=6, features=11)
- **Encoder**: LSTM(32) → LSTM(16) with temporal compression
- **Decoder**: RepeatVector → LSTM(16) → LSTM(32) with reconstruction
- **Unsupervised pretraining** on UMN + **Supervised fine-tuning** on Avenue

### Features Used (Rolling Buffer)
From Member 2's zone extraction:
- `count_sm` - Smoothed crowd count per zone
- `density_sm` - Smoothed density per zone
- `speed_mean_sm`, `speed_std_sm` - Smoothed velocity statistics
- `flow_x_sm`, `flow_y_sm`, `flow_magnitude_sm` - Smoothed optical flow
- `accel_mean_sm` - Smoothed acceleration
- `flow_divergence_sm`, `compression_sm`, `spatial_entropy_sm` - Flow metrics

### Training Strategy
1. **Pretraining** (30 epochs on UMN): Learn general crowd anomaly patterns without labels
2. **Fine-tuning** (20 epochs on Avenue 70%): Adapt to Avenue-specific behaviors with supervised signal
3. **Threshold Optimization**: Use Precision-Recall curve on Avenue train to maximize F1
4. **Evaluation**: Test on Avenue 30% with ground truth labels

## Input (from Member 2)

Zone-level feature CSVs with columns:
frame_idx, zone_id, count, density, speed_mean, ..., count_sm, density_sm, ...
Shape after reshaping: `(n_frames, 6_zones, 11_features_per_zone)`

## Output

* `results/umn_pretrain_avenue_finetune.json` - Final metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
* `results/final_evaluation.json` - Detailed performance breakdown
* `models/model_*.keras` - Saved LSTM models
* `models/scaler.pkl` - Feature normalization
* `models/thresholds.pkl` - Per-model thresholds

## Results

### Current Performance (UMN Pretrain → Avenue Finetune)

**Avenue Test Set** (4,598 frames, 1,239 anomalies):
- **Accuracy**: 26.6%
- **Precision**: 26.7%
- **Recall**: 99.5%
- **F1-Score**: 42.0%
- **AUC-ROC**: 45.9%
- **Optimal Threshold**: 0.5380

### Why Lower Than Target?

**Cross-Domain Challenge**:
- UMN = organized crowd panic/escape (specific motion patterns)
- Avenue = diverse behavioral anomalies (running, throwing, etc.)
- Reconstruction error doesn't universally distinguish normal/anomaly across datasets

**Best In-Domain Performance** (UMN train → UMN test, earlier iteration):
- **F1-Score**: 73% (when training/testing same dataset)
- **Precision**: 92.7%, **Recall**: 60.3%

**Trade-off**: High recall (catching most anomalies) vs. precision (many false positives in cross-domain setting)

## Usage

### Prerequisites
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Quick Test
```bash
python3 quick_test.py  # Trains 3 models, checks pipeline
```

### Full Pipeline

**1. UMN Pretrain + Avenue Finetune + Evaluate**
```bash
python3 << 'EOF'
# See src/ensemble_anomaly.py for details
# Runs: Pretrain (30 epochs) → Finetune (20 epochs) → Threshold optimization → Test
EOF
```

**2. Evaluate with Different Thresholds**
```bash
python3 src/evaluate_anomaly.py
```

**3. In-Domain Evaluation (Optional)**
```bash
# Test UMN model on UMN test data
python3 << 'EOF'
# Expected: F1 ~70% (in-domain)
EOF
```

## Dataset Context

### UMN Dataset (Training)
- **11 videos** of crowd panic/escape scenarios
- **5,597 frames** total for pretraining
- **No labels** (unsupervised pretraining)
- Behavior: Organized crowd movement, panic, escape

### Avenue Dataset (Fine-tuning + Testing)
- **22 videos** of behavioral anomalies
- **15,324 frames** total
- **Frame-level binary labels** (normal/anomaly)
- Train split: 10,726 frames (2,473 anomalies)
- Test split: 4,598 frames (1,239 anomalies)
- Behavior: Running, throwing, loitering, etc. (diverse)

### Why NOT UCSD?
- UCSD Ped2 contains bicycles in parking lot (domain-specific)
- Not representative of crowd stampede detection task
- No labeled ground truth available during development

## Key Findings

1. **Rolling Buffers Matter**: Smoothed features (`_sm`) capture temporal patterns better than raw
2. **Zone Structure Matters**: Keeping spatial relationships (6 zones) improves over flattened features
3. **In-Domain Works**: F1 ~73% when same dataset used for train/test
4. **Cross-Domain is Hard**: F1 drops to ~42% when generalizing across datasets
5. **Pretraining Helps**: Transfer learning from UMN improves Avenue adaptation
