import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix

from pipeline import run_pipeline


def calculate_iou(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0


def main():
    results_df, _, lstm_results = run_pipeline()

    y_true, y_pred_risk, y_pred_combined = [], [], []

    for _, row in results_df.iterrows():
        vid_id, frame_num = row['vid_id'], row['frame']
        gt_anomaly_frames = set(lstm_results.get(vid_id, {}).get('anomaly_frames', []))
        y_true.append(1 if frame_num in gt_anomaly_frames else 0)
        y_pred_risk.append(1 if row['risk_level'] in ['HIGH', 'CRITICAL'] else 0)
        y_pred_combined.append(1 if row['alert_source'] != 'NONE' else 0)

    for label, y_pred in [("RISK-BASED", y_pred_risk), ("COMBINED", y_pred_combined)]:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        iou   = calculate_iou(y_true, y_pred)
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        print(f"Accuracy:      {acc:.4f}  ({acc*100:.2f}%)")
        print(f"Precision:     {prec:.4f}")
        print(f"Recall:        {rec:.4f}")
        print(f"F1-Score:      {f1:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"IoU:           {iou:.4f}")

    metrics = {
        'risk_based': {'accuracy': float(accuracy_score(y_true, y_pred_risk))},
        'combined':   {'accuracy': float(accuracy_score(y_true, y_pred_combined))}
    }
    with open('risk_assessment_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved risk_assessment_metrics.json")


if __name__ == '__main__':
    main()