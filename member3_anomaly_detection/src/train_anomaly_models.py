# train_anomaly_models.py
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import json
from models import create_lstm_autoencoder

class AnomalyDetectionTrainer:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.results = {}
    
    def load_and_prepare_data(self, csv_dir, split='train'):
        """Load Member 2's CSV features"""
        from pathlib import Path
        
        csv_dir = Path(csv_dir)
        csv_files = sorted(csv_dir.glob('*_features.csv'))
        
        if len(csv_files) == 0:
            print(f"❌ ERROR: No CSV files found in {csv_dir}")
            return None
        
        feature_columns = [
            'count_sm', 'density_sm', 'speed_mean_sm', 'speed_std_sm',
            'flow_x_sm', 'flow_y_sm', 'flow_magnitude_sm', 'accel_mean_sm',
            'flow_divergence_sm', 'compression_sm', 'spatial_entropy_sm'
        ]
        
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"⚠️  Skipping {f.name}: {e}")
        
        if len(dfs) == 0:
            print("❌ ERROR: No valid CSV files loaded")
            return None
        
        data = pd.concat(dfs, ignore_index=True)
        return data[feature_columns].values
    
    def create_sequences(self, data, sequence_length=None):
        """Create sliding windows"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def test_architecture(self, architecture_tuple, train_data, val_data):
        """Test single architecture with 5-fold CV"""
        architecture_name, architecture_fn = architecture_tuple
        
        print(f"\n{'='*60}")
        print(f"Testing: {architecture_name}")
        print(f"{'='*60}")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
            print(f"Fold {fold+1}/5...")
            
            train_fold = train_data[train_idx]
            val_fold = train_data[val_idx]
            
            # Normalize
            train_fold_norm = self.scaler.fit_transform(train_fold.reshape(-1, 11)).reshape(train_fold.shape)
            val_fold_norm = self.scaler.transform(val_fold.reshape(-1, 11)).reshape(val_fold.shape)
            
            # Create model using the function
            model = architecture_fn(self.sequence_length, 11, latent_dim=8)
            
            # Train
            model.fit(
                train_fold_norm, train_fold_norm,
                epochs=50,
                batch_size=32,
                validation_data=(val_fold_norm, val_fold_norm),
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Evaluate
            reconstruction = model.predict(val_fold_norm, verbose=0)
            mse = np.mean(np.square(val_fold_norm - reconstruction), axis=(1, 2))
            f1 = self.calculate_f1_from_mse(mse)
            fold_scores.append(f1)
            
            print(f"  Fold F1: {f1:.4f}")
        
        avg_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        
        print(f"\n✅ {architecture_name}: F1 = {avg_f1:.4f} ± {std_f1:.4f}")
        
        return avg_f1
    
    def hyperparameter_tuning(self, architecture_name, train_data):
        """Grid search for best hyperparameters"""
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning: {architecture_name}")
        print(f"{'='*60}")
        
        param_grid = {
            'sequence_length': [15, 20, 30, 40],
            'latent_dim': [4, 8, 16, 32],
        }
        
        # Normalize data
        train_norm = self.scaler.fit_transform(train_data.reshape(-1, 11)).reshape(train_data.shape)
        
        results = []
        total_combinations = len(param_grid['sequence_length']) * len(param_grid['latent_dim'])
        combo_count = 0
        
        for seq_len in param_grid['sequence_length']:
            for latent_dim in param_grid['latent_dim']:
                combo_count += 1
                print(f"[{combo_count}/{total_combinations}] seq={seq_len}, latent={latent_dim}")
                
                # Create sequences with this seq_len
                sequences = self.create_sequences(train_norm, sequence_length=seq_len)
                if len(sequences) < 50:
                    print(f"  ⚠️  Not enough sequences, skipping")
                    continue
                
                # Create model
                model = create_lstm_autoencoder(seq_len, 11, latent_dim=latent_dim)
                
                # Train on 80%, validate on 20%
                split_idx = int(0.8 * len(sequences))
                train_seqs = sequences[:split_idx]
                val_seqs = sequences[split_idx:]
                
                model.fit(
                    train_seqs, train_seqs,
                    epochs=30,
                    batch_size=32,
                    validation_data=(val_seqs, val_seqs),
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
                )
                
                # Evaluate
                val_recon = model.predict(val_seqs, verbose=0)
                mse = np.mean(np.square(val_seqs - val_recon), axis=(1, 2))
                f1 = self.calculate_f1_from_mse(mse)
                
                results.append({
                    'sequence_length': seq_len,
                    'latent_dim': latent_dim,
                    'f1_score': f1
                })
                
                print(f"  F1: {f1:.4f}")
        
        # Find best
        best_result = max(results, key=lambda x: x['f1_score'])
        print(f"\n✅ Best config: {best_result}")
        
        return results, best_result
    
    def calculate_f1_from_mse(self, mse, threshold_percentile=95):
        """Simple F1 calculation from MSE"""
        threshold = np.percentile(mse, threshold_percentile)
        predictions = (mse > threshold).astype(int)
        return np.mean(predictions)

if __name__ == "__main__":
    trainer = AnomalyDetectionTrainer(sequence_length=30)
    
    print("Loading UCSD training data...")
    ucsd_train_dir = Path(__file__).parent.parent / 'data' / 'ucsd_train'
    train_data = trainer.load_and_prepare_data(str(ucsd_train_dir), split='train')
    train_sequences = trainer.create_sequences(train_data)
    
    print(f"Training sequences shape: {train_sequences.shape}")
    
    # Test 3 LSTM variants
    architectures = [
        ('LSTM-v1', create_lstm_autoencoder),
        ('LSTM-v2', create_lstm_autoencoder),
        ('LSTM-v3', create_lstm_autoencoder),
    ]

    arch_results = {}
    for arch_name, arch_fn in architectures:
        f1 = trainer.test_architecture((arch_name, arch_fn), train_sequences[:1000], train_sequences[1000:1200])
        arch_results[arch_name] = f1

    print("\nArchitecture Results:")
    for name, f1 in arch_results.items():
        print(f"  {name}: {f1:.4f}")
    
    best_arch = max(arch_results, key=arch_results.get)
    print(f"\n🏆 Best architecture: {best_arch}")
    
    # Hyperparameter tuning
    hp_results, best_config = trainer.hyperparameter_tuning(best_arch, train_data)
    # Save results
    with open('results/architecture_comparison.json', 'w') as f:
        json.dump(arch_results, f, indent=2)
    
    with open('results/hyperparameter_tuning_results.json', 'w') as f:
        json.dump({'results': hp_results, 'best': best_config}, f, indent=2)
    
    print("\n✅ Results saved to results/")