import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import THRESHOLDS, FRAME_WIDTH, FRAME_HEIGHT, GRID_ROWS, GRID_COLS, CELL_WIDTH
from data_loader import load_features, load_lstm_results, build_lstm_scores
from spatial_analyzer import SpatialAnalyzer
from risk_scorer import RiskScorer


def run_pipeline(features_dir='features', results_path='results.json'):
    features_dict = load_features(features_dir)
    lstm_results  = load_lstm_results(results_path)
    lstm_scores_dict = build_lstm_scores(lstm_results)

    spatial_analyzer = SpatialAnalyzer(FRAME_WIDTH, FRAME_HEIGHT, GRID_ROWS, GRID_COLS)
    risk_scorer = RiskScorer()

    all_results = []
    video_summaries = {}
    common_vids = set(features_dict.keys()) & set(lstm_scores_dict.keys())
    print(f"Processing {len(common_vids)} videos...\n")

    for vid_id in tqdm(sorted(common_vids)):
        features_df  = features_dict[vid_id]
        lstm_scores  = lstm_scores_dict[vid_id]
        frame_results = []
        alerts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        lstm_anomaly_frames, spatial_anomaly_frames, multi_source_validations = [], [], []

        for idx, row in features_df.iterrows():
            frame_num    = int(row['frame'])
            lstm_score   = lstm_scores[frame_num] if frame_num < len(lstm_scores) else 0.0
            person_count = int(row.get('total_count_smooth', row.get('total_count', 0)))

            left_count   = int(row.get('left_smooth',   row.get('left',   0)))
            center_count = int(row.get('center_smooth', row.get('center', 0)))
            right_count  = int(row.get('right_smooth',  row.get('right',  0)))

            bboxes = []
            for i in range(left_count):
                x1 = np.random.randint(0, CELL_WIDTH)
                y1 = np.random.randint(0, FRAME_HEIGHT)
                w, h = np.random.randint(40, 80), np.random.randint(80, 150)
                bboxes.append([x1, y1, x1+w, y1+h, 0.9])
            for i in range(center_count):
                x1 = np.random.randint(CELL_WIDTH, 2*CELL_WIDTH)
                y1 = np.random.randint(0, FRAME_HEIGHT)
                w, h = np.random.randint(40, 80), np.random.randint(80, 150)
                bboxes.append([x1, y1, x1+w, y1+h, 0.9])
            for i in range(right_count):
                x1 = np.random.randint(2*CELL_WIDTH, FRAME_WIDTH)
                y1 = np.random.randint(0, FRAME_HEIGHT)
                w, h = np.random.randint(40, 80), np.random.randint(80, 150)
                bboxes.append([x1, y1, x1+w, y1+h, 0.9])

            spatial_anomalies, spatial_score = spatial_analyzer.analyze_frame(bboxes, row)
            risk_score, risk_components = risk_scorer.compute_risk_score(lstm_score, spatial_score, person_count)
            risk_level = risk_scorer.classify_risk_level(risk_score)
            alerts[risk_level] += 1

            lstm_anomaly    = lstm_score > THRESHOLDS['lstm_anomaly']
            spatial_anomaly = spatial_score > 0.3
            if lstm_anomaly:    lstm_anomaly_frames.append(frame_num)
            if spatial_anomaly: spatial_anomaly_frames.append(frame_num)
            if lstm_anomaly and spatial_anomaly:
                multi_source_validations.append(frame_num)

            alert_source = ('BOTH_M3_M4'       if lstm_anomaly and spatial_anomaly else
                            'MEMBER3_LSTM'      if lstm_anomaly else
                            'MEMBER4_SPATIAL'   if spatial_anomaly else 'NONE')

            frame_result = {
                'vid_id': vid_id, 'frame': frame_num,
                'person_count': person_count,
                'lstm_score': lstm_score, 'spatial_score': spatial_score,
                'risk_score': risk_score, 'risk_level': risk_level,
                'alert_source': alert_source,
                'crawling': spatial_anomalies['crawling'],
                'overcrowding': spatial_anomalies['overcrowding'],
                'zone_imbalance': spatial_anomalies['zone_imbalance'],
                'lstm_anomaly': lstm_anomaly, 'spatial_anomaly': spatial_anomaly,
                **risk_components
            }
            frame_results.append(frame_result)
            all_results.append(frame_result)

        total = len(frame_results)
        video_summaries[vid_id] = {
            'total_frames': total, 'alerts': alerts,
            'lstm_anomaly_frames': len(lstm_anomaly_frames),
            'spatial_anomaly_frames': len(spatial_anomaly_frames),
            'multi_source_validations': len(multi_source_validations),
            'multi_source_rate': len(multi_source_validations) / total if total else 0,
            'avg_risk_score': np.mean([r['risk_score'] for r in frame_results]),
            'max_risk_score': np.max([r['risk_score'] for r in frame_results]),
            'critical_rate': alerts['CRITICAL'] / total if total else 0
        }

    results_df = pd.DataFrame(all_results)
    return results_df, video_summaries, lstm_results


if __name__ == '__main__':
    results_df, video_summaries, _ = run_pipeline()
    results_df.to_csv('risk_assessment_detailed.csv', index=False)
    pd.DataFrame(video_summaries).T.to_csv('risk_assessment_summary.csv')
    alerts_df = results_df[results_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
    alerts_df[['vid_id','frame','risk_score','risk_level','alert_source',
               'person_count','crawling','overcrowding','zone_imbalance']].to_csv('alerts_log.csv', index=False)
    print(f"\nDone. {len(results_df)} frames, {len(alerts_df)} alerts.")