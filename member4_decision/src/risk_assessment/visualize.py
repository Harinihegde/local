import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from pipeline import run_pipeline
from config import THRESHOLDS, RISK_LEVELS, DENSITY_THRESHOLDS


def main():
    results_df, _, lstm_results = run_pipeline()

    y_true      = []
    y_pred_risk = []
    for _, row in results_df.iterrows():
        gt = set(lstm_results.get(row['vid_id'], {}).get('anomaly_frames', []))
        y_true.append(1 if row['frame'] in gt else 0)
        y_pred_risk.append(1 if row['risk_level'] in ['HIGH', 'CRITICAL'] else 0)

    # 1. Risk distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
    risk_counts = results_df['risk_level'].value_counts()
    risk_counts.plot(kind='bar', ax=axes[0], color=[colors.get(x, 'gray') for x in risk_counts.index])
    axes[0].set_title('Risk Level Distribution'); axes[0].grid(axis='y', alpha=0.3)
    results_df['alert_source'].value_counts().plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Alert Source Distribution'); axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig('risk_distribution.png', dpi=150); plt.show()

    # 2. Component score distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0,0].hist(results_df['lstm'],    bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[0,0].axvline(THRESHOLDS['lstm_anomaly'], color='red', linestyle='--'); axes[0,0].set_title('LSTM Score (40%)')
    axes[0,1].hist(results_df['spatial'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(0.3, color='red', linestyle='--'); axes[0,1].set_title('Spatial Score (30%)')
    axes[1,0].hist(results_df['density'], bins=50, color='green',  alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Density Score (20%)')
    axes[1,1].hist(results_df['risk_score'], bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Final Risk Score')
    plt.tight_layout(); plt.savefig('component_analysis.png', dpi=150); plt.show()

    # 3. Confusion matrix
    cm = confusion_matrix(y_true, y_pred_risk)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix'); plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150); plt.show()

    # 4. Sample timeline
    sample_vid  = results_df['vid_id'].iloc[0]
    sample_data = results_df[results_df['vid_id'] == sample_vid].sort_values('frame')
    frames      = sample_data['frame']

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(frames, sample_data['risk_score'], color='red', linewidth=2)
    axes[0].fill_between(frames, 0, sample_data['risk_score'], alpha=0.3, color='red')
    axes[0].set_title(f'Risk Score Timeline - {sample_vid}'); axes[0].grid(alpha=0.3)
    for comp, label in [('lstm','LSTM(40%)'),('spatial','Spatial(30%)'),('density','Density(20%)'),('size','Size(10%)')]:
        axes[1].plot(frames, sample_data[comp], label=label, linewidth=2)
    axes[1].legend(); axes[1].set_title('Component Contributions'); axes[1].grid(alpha=0.3)
    axes[2].plot(frames, sample_data['person_count'], color='steelblue', linewidth=2)
    axes[2].axhline(DENSITY_THRESHOLDS['sparse'][1], color='green', linestyle='--', label='Sparse/Medium')
    axes[2].axhline(DENSITY_THRESHOLDS['medium'][1], color='orange', linestyle='--', label='Medium/Dense')
    axes[2].legend(); axes[2].set_title('Crowd Size'); axes[2].set_xlabel('Frame'); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('sample_timeline.png', dpi=150); plt.show()

    print("All visualizations saved.")


if __name__ == '__main__':
    main()