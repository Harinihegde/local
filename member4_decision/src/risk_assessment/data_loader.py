import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


def load_features(features_dir='features'):
    feature_files = sorted(Path(features_dir).glob('*.csv'))
    feature_files = [f for f in feature_files if 'dataset_summary' not in f.name]
    features_dict = {}
    for csv_path in tqdm(feature_files, desc="Loading features"):
        parts = csv_path.stem.split('_')
        vid_id = parts[1] if len(parts) >= 2 else csv_path.stem
        features_dict[vid_id] = pd.read_csv(csv_path)
    print(f"Loaded features for {len(features_dict)} videos")
    return features_dict


def load_lstm_results(results_path='results.json'):
    with open(results_path, 'r') as f:
        lstm_results = json.load(f)
    print(f"Loaded LSTM results for {len(lstm_results)} videos")
    return lstm_results


def build_lstm_scores(lstm_results):
    lstm_scores_dict = {}
    for vid_id, data in lstm_results.items():
        anomaly_scores = data.get('anomaly_scores', [])
        if len(anomaly_scores) > 0:
            lstm_scores_dict[vid_id] = anomaly_scores
        else:
            anomaly_frames = set(data.get('anomaly_frames', []))
            total_frames = data['total_frames']
            lstm_scores_dict[vid_id] = [
                1.0 if i in anomaly_frames else 0.0
                for i in range(total_frames)
            ]
    return lstm_scores_dict