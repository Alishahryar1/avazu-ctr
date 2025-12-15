import polars as pl
import numpy as np
import gc
from tqdm import tqdm
from config import CONFIG
import os
import pickle

def process_data_polars():
    print("Loading data with Polars (Schema Fixed)...")
    
    # --- 1. Define Schema to prevent Parsing Errors ---
    # We force 'id' to String to handle the massive integers.
    # We force other categoricals to String initially to speed up parsing.
    dtypes = {
        'id': pl.String,
        'click': pl.UInt8,
        'hour': pl.String,
        'C1': pl.String,
        'banner_pos': pl.String,
        'site_id': pl.String,
        'site_domain': pl.String,
        'site_category': pl.String,
        'app_id': pl.String,
        'app_domain': pl.String,
        'app_category': pl.String,
        'device_id': pl.String,
        'device_ip': pl.String,
        'device_model': pl.String,
        'device_type': pl.String,
        'device_conn_type': pl.String,
        'C14': pl.String,
        'C15': pl.String,
        'C16': pl.String,
        'C17': pl.String,
        'C18': pl.String,
        'C19': pl.String,
        'C20': pl.String,
        'C21': pl.String,
    }
    
    # Columns to actually train on
    cat_cols = [
        'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
        'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]
    
    # Lazy load with explicit dtypes
    q_train = pl.scan_csv(CONFIG['train_path'], schema_overrides=dtypes)
    q_test = pl.scan_csv(CONFIG['test_path'], schema_overrides=dtypes)
    
    # --- 2. Feature Engineering (Time) ---
    def extract_time_features(q):
        return q.with_columns([
            pl.col("hour").str.slice(6, 2).cast(pl.UInt8).alias("hour_of_day"),
            pl.col("hour").str.slice(4, 2).cast(pl.UInt8).alias("day_of_week"),
        ])

    q_train = extract_time_features(q_train)
    q_test = extract_time_features(q_test)
    
    # Add new time features to the list of categorical columns
    cat_cols += ['hour_of_day', 'day_of_week']

    print("Building vocabularies (Frequency Thresholding)...")
    
    vocab_sizes = {}
    feat_maps = {}
    
    # We iterate to save memory, but with 30GB you could potentially do groups.
    # Iteration is safer to avoid OOM spikes during the 'collect'.
    for col in tqdm(cat_cols, desc="Building Vocabs"):
        # 1. Count frequencies in Train
        # We cast to String to ensure matching types between Train/Test
        counts = q_train.select(pl.col(col).cast(pl.String)).group_by(col).len().collect()
        
        # 2. Filter: Keep only features appearing >= min_freq
        frequent_items = counts.filter(pl.col("len") >= CONFIG['min_freq'])[col].to_list()
        frequent_items.sort() # Ensure deterministic order for mapping

        
        # 3. Create Map: Value -> Int ID (Start at 1, 0 is <UNK>)
        # Using a dictionary is fast for Polars 'replace'
        mapping = {val: i + 1 for i, val in enumerate(frequent_items)}
        
        feat_maps[col] = mapping
        vocab_sizes[col] = len(mapping) + 1
        
    print("Vocabularies built. Mapping and converting to Numpy...")

    # --- 3. Transformation Function ---
    def map_and_convert(df_lazy, is_test=False):
        # Select only necessary columns to save RAM
        # Include 'hour' for temporal splitting (train only)
        cols_to_select = cat_cols + (['id'] if is_test else ['click', 'hour'])
        
        # Materialize the dataframe using gpu engine.
        df = df_lazy.select(cols_to_select).collect(engine="gpu")
        
        # Extract ID or Target before mapping
        extra_data = df['id'].to_numpy() if is_test else df['click'].to_numpy().astype(np.float32)
        
        # Extract raw hour for temporal splitting (train only)
        hour_data = None if is_test else df['hour'].to_numpy()
        
        # Perform mapping in-place (or close to it)
        # We loop through columns to map them to integers
        for col in tqdm(cat_cols, desc=f"Mapping {'Test' if is_test else 'Train'}"):
            mapping = feat_maps[col]
            
            # Polars 'replace' is very optimized. 
            # Values not in the mapping (rare or new) become null, we fill with 0 (<UNK>)
            df = df.with_columns(
                pl.col(col).cast(pl.String) # Ensure type match
                .replace_strict(mapping, default=0)
                .cast(pl.Int32)
            )
        
        # Convert the feature matrix to Numpy
        X_data = df.select(cat_cols).to_numpy().astype(np.int32)
        
        # Clean up Polars DF to free RAM immediately
        del df
        gc.collect()
        
        return X_data, extra_data, hour_data

    # Process Train (returns hour data for temporal splitting)
    X_train, y_train, train_hours = map_and_convert(q_train, is_test=False)
    print(f"Train processed. Shape: {X_train.shape}")
    
    # Process Test (hour_data is None for test)
    X_test, test_ids, _ = map_and_convert(q_test, is_test=True)
    print(f"Test processed. Shape: {X_test.shape}")
    
    return X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols

def save_processed_data(X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, feature_names):
    """Saves processed data to disk including hour data for temporal splitting."""
    print(f"Saving processed data to {CONFIG['processed_path']}...")
    os.makedirs(CONFIG['processed_path'], exist_ok=True)
    
    np.save(os.path.join(CONFIG['processed_path'], "X_train.npy"), X_train)
    np.save(os.path.join(CONFIG['processed_path'], "y_train.npy"), y_train)
    np.save(os.path.join(CONFIG['processed_path'], "train_hours.npy"), train_hours)  # For temporal split
    np.save(os.path.join(CONFIG['processed_path'], "X_test.npy"), X_test)
    np.save(os.path.join(CONFIG['processed_path'], "test_ids.npy"), test_ids)
    
    import pickle
    with open(os.path.join(CONFIG['processed_path'], "vocab_sizes.pkl"), "wb") as f:
        pickle.dump(vocab_sizes, f)
        
    with open(os.path.join(CONFIG['processed_path'], "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
        
    print("Data saved successfully.")

def load_processed_data(mode: str = 'train') -> tuple[
    np.ndarray | None,  # X_train
    np.ndarray | None,  # y_train
    np.ndarray | None,  # train_hours (for temporal split)
    np.ndarray,         # X_test
    np.ndarray,         # test_ids
    dict,               # vocab_sizes
    list                # feature_names
]:
    """
    Loads processed data from disk.
    mode: 'train' (loads everything including hours for temporal split), 'inference' (loads only test data and metadata)
    """
    path = CONFIG['processed_path']
    print(f"Loading processed data from {path} for {mode}...")
    
    try:
        with open(os.path.join(path, "vocab_sizes.pkl"), "rb") as f:
            vocab_sizes = pickle.load(f)
        with open(os.path.join(path, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
            
        if mode == 'train':
            X_train = np.load(os.path.join(path, "X_train.npy"), allow_pickle=True)
            y_train = np.load(os.path.join(path, "y_train.npy"), allow_pickle=True)
            train_hours = np.load(os.path.join(path, "train_hours.npy"), allow_pickle=True)
            X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
            test_ids = np.load(os.path.join(path, "test_ids.npy"), allow_pickle=True)
            return X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, feature_names
        
        elif mode == 'inference':
            X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
            test_ids = np.load(os.path.join(path, "test_ids.npy"), allow_pickle=True)
            return None, None, None, X_test, test_ids, vocab_sizes, feature_names
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'.")
            
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Processed data not found in {path}. Please run 'python data_processor.py' first.") from e

if __name__ == "__main__":
    X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols = process_data_polars()
    save_processed_data(X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols)

