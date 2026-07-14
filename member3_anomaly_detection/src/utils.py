import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_csv_data(csv_dir, split='train'):
    """Load Member 2's CSV features"""
    csv_dir = Path(csv_dir)
    
    csv_files = sorted(csv_dir.glob('*_features.csv'))
    
    if len(csv_files) == 0:
        print(f"❌ ERROR: No CSV files found in {csv_dir}")
        return None
    
    # Use smoothed features from Member 2's output
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
            else:
                print(f"⚠️  Skipping empty file: {f.name}")
        except Exception as e:
            print(f"⚠️  Error reading {f.name}: {e}")
    
    if len(dfs) == 0:
        print("❌ ERROR: No valid CSV files loaded")
        return None
    
    data = pd.concat(dfs, ignore_index=True)
    
    return data[feature_columns].values
def create_sequences(data, sequence_length=30):
    """Create sliding windows"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)