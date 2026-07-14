"""
Quick test: Train ensemble and evaluate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.train_anomaly_models import AnomalyDetectionTrainer
from src.ensemble_anomaly import AnomalyEnsemble
from src.evaluate_anomaly import AnomalyEvaluator
from src.utils import load_csv_data
import json

if __name__ == "__main__":
    print("="*60)
    print("MEMBER 3: ANOMALY DETECTION ENSEMBLE")
    print("="*60)
    
    # Define paths
    ucsd_train_dir = Path(__file__).parent / 'data' / 'ucsd_train'
    
    # Load UCSD train data
    print("\n1. Loading UCSD training data...")
    train_data = load_csv_data(str(ucsd_train_dir))
    print(f"   ✅ Loaded {len(train_data)} frames")
    
    # Train ensemble
    print("\n2. Training ensemble (3 models)...")
    ensemble = AnomalyEnsemble(sequence_length=30)
    ensemble.train_ensemble(train_data, n_models=3)
    
    print("\n✅ QUICK TEST COMPLETE!")
    print("   Next: Run full evaluation on UCSD + Avenue test sets")