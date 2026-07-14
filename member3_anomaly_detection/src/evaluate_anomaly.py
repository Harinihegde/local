import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class AnomalyEvaluator:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.models = []
        self.thresholds = []
        self.scaler = None
    
    def load_models(self, model_dir):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(model_dir / 'thresholds.pkl', 'rb') as f:
            self.thresholds = pickle.load(f)
        
        for i in range(3):
            model = tf.keras.models.load_model(model_dir / f'model_{i}.keras')
            self.models.append(model)
        
        print(f"✅ Loaded {len(self.models)} models")
    
    def evaluate_with_labels(self, test_data, ground_truth):
        """Evaluate with ground truth labels"""
        sequences = []
        for i in range(len(test_data) - self.sequence_length + 1):
            sequences.append(test_data[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        sequences_norm = self.scaler.transform(sequences.reshape(-1, 66)).reshape(sequences.shape)
        
        # Get ensemble predictions
        all_scores = []
        for model in self.models:
            recon = model.predict(sequences_norm, verbose=0)
            mse = np.mean(np.square(sequences_norm - recon), axis=(1, 2))
            all_scores.append(mse)
        
        all_scores = np.array(all_scores)
        ensemble_score = np.mean(all_scores, axis=0)
        
        # Threshold on training data (95th percentile)
        threshold = np.percentile(ensemble_score, 95)
        predictions = (ensemble_score > threshold).astype(int)
        
        # Align with ground truth
        gt = ground_truth[:len(predictions)].values
        
        # Calculate metrics
        accuracy = accuracy_score(gt, predictions)
        precision = precision_score(gt, predictions, zero_division=0)
        recall = recall_score(gt, predictions, zero_division=0)
        f1 = f1_score(gt, predictions, zero_division=0)
        kappa = cohen_kappa_score(gt, predictions)
        auc = roc_auc_score(gt, ensemble_score)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cohen_kappa': float(kappa),
            'auc_roc': float(auc),
            'threshold': float(threshold),
            'total_frames': len(predictions),
            'true_positives': int(np.sum((predictions == 1) & (gt == 1))),
            'false_positives': int(np.sum((predictions == 1) & (gt == 0))),
            'true_negatives': int(np.sum((predictions == 0) & (gt == 0))),
            'false_negatives': int(np.sum((predictions == 0) & (gt == 1)))
        }
        
        return results

if __name__ == "__main__":
    print("="*60)
    print("MEMBER 3: ANOMALY DETECTION EVALUATION (WITH LABELS)")
    print("="*60)
    
    evaluator = AnomalyEvaluator(sequence_length=30)
    evaluator.load_models('models/')
    
    # Load UMN reshaped data (for training the scaler)
    umn_train_scene1 = pd.read_csv('data/umn_train/scene1_reshaped.csv').values
    umn_train_scene2 = pd.read_csv('data/umn_train/scene2_reshaped.csv').values
    umn_train = np.vstack([umn_train_scene1, umn_train_scene2])
    
    # Load Avenue test data
    avenue_test_scene1 = pd.read_csv('data/avenue_test/scene1_reshaped.csv').values if Path('data/avenue_test/scene1_reshaped.csv').exists() else None
    
    # Load labels
    labels = pd.read_csv('data/avenue_test_labels.csv')
    
    print(f"\nLoaded {len(umn_train)} UMN train frames")
    print(f"Loaded {len(labels)} Avenue test frames with labels")
    print(f"Anomalies: {labels['ground_truth'].sum()} ({100*labels['ground_truth'].mean():.1f}%)")
    
    # Since we don't have reshaped Avenue data yet, test on UMN
    # (In practice, you'd also have Avenue reshaped)
    print("\n" + "="*60)
    print("Evaluating on Avenue Test (with ground truth)")
    print("="*60)
    
    # For now, evaluate on a subset (you'll need Avenue reshaped data)
    results = {
        'status': 'Ready for evaluation',
        'note': 'Waiting for Avenue reshaped data',
        'labels_loaded': len(labels),
        'anomalies': int(labels['ground_truth'].sum())
    }
    
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Ready to evaluate!")
