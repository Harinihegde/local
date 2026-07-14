# ensemble_anomaly.py
import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from models import create_lstm_autoencoder

class AnomalyEnsemble:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = []
        self.thresholds = []
    
    def train_ensemble(self, train_data, n_models=3):
        """Train multiple models with different seeds"""
        print("\n" + "="*60)
        print(f"Training Ensemble ({n_models} models)")
        print("="*60)
        
        # Create sequences
        sequences = []
        for i in range(len(train_data) - self.sequence_length + 1):
            sequences.append(train_data[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        # Normalize
        sequences_norm = self.scaler.fit_transform(sequences.reshape(-1, 11)).reshape(sequences.shape)
        
        for idx in range(n_models):
            print(f"\nModel {idx+1}/{n_models}")
            
            # Set different seed for each
            np.random.seed(idx)
            tf.random.set_seed(idx)
            
            # Create and train
            model = create_lstm_autoencoder(self.sequence_length, 11, latent_dim=8)
            model.fit(
                sequences_norm, sequences_norm,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Calculate threshold
            recon = model.predict(sequences_norm, verbose=0)
            mse = np.mean(np.square(sequences_norm - recon), axis=(1, 2))
            threshold = np.percentile(mse, 95)
            
            self.models.append(model)
            self.thresholds.append(threshold)
            
            print(f"  ✅ Threshold: {threshold:.6f}")
        
        return self.models
    
    def predict_ensemble(self, test_data):
        """Predict on test data using ensemble"""
        # Create sequences
        sequences = []
        for i in range(len(test_data) - self.sequence_length + 1):
            sequences.append(test_data[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        # Normalize
        sequences_norm = self.scaler.transform(sequences.reshape(-1, 11)).reshape(sequences.shape)
        
        # Get predictions from all models
        all_scores = []
        all_predictions = []
        
        for model_idx, model in enumerate(self.models):
            recon = model.predict(sequences_norm, verbose=0)
            mse = np.mean(np.square(sequences_norm - recon), axis=(1, 2))
            
            # Normalize to [0, 1]
            mse_norm = (mse - np.min(mse)) / (np.max(mse) - np.min(mse) + 1e-6)
            all_scores.append(mse_norm)
            
            # Binary prediction
            pred = (mse > self.thresholds[model_idx]).astype(int)
            all_predictions.append(pred)
        
        all_scores = np.array(all_scores)
        all_predictions = np.array(all_predictions)
        
        # Ensemble: average score + majority voting
        ensemble_score = np.mean(all_scores, axis=0)
        ensemble_pred = np.sum(all_predictions, axis=0) >= (len(self.models) // 2 + 1)
        
        return ensemble_score, ensemble_pred.astype(int), all_scores
    
    def save_models(self, save_dir):
        """Save trained models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, model in enumerate(self.models):
            model.save(save_dir / f'model_{idx}.keras')
        
        with open(save_dir / 'thresholds.pkl', 'wb') as f:
            pickle.dump(self.thresholds, f)
        
        with open(save_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✅ Models saved to {save_dir}")

if __name__ == "__main__":
    from pathlib import Path
    
    print("="*60)
    print("MEMBER 3: ANOMALY DETECTION ENSEMBLE TRAINING")
    print("="*60)
    
    # Load training data
    print("\n1. Loading UCSD training data...")
    ucsd_train_dir = Path(__file__).parent.parent / 'data' / 'ucsd_train'
    
    feature_columns = [
        'count_sm', 'density_sm', 'speed_mean_sm', 'speed_std_sm',
        'flow_x_sm', 'flow_y_sm', 'flow_magnitude_sm', 'accel_mean_sm',
        'flow_divergence_sm', 'compression_sm', 'spatial_entropy_sm'
    ]
    
    csv_files = sorted(ucsd_train_dir.glob('*_features.csv'))
    dfs = [pd.read_csv(f) for f in csv_files]
    train_data = pd.concat(dfs, ignore_index=True)
    train_data = train_data[feature_columns].values
    
    print(f"   ✅ Loaded {len(train_data)} frames")
    
    # Train ensemble
    print("\n2. Training ensemble (3 models)...")
    ensemble = AnomalyEnsemble(sequence_length=30)
    ensemble.train_ensemble(train_data, n_models=3)
    
    # Save models
    print("\n3. Saving models...")
    ensemble.save_models('models/')
    
    print("\n✅ ENSEMBLE TRAINING COMPLETE!")
    print("   Next: Run src/evaluate_anomaly.py")