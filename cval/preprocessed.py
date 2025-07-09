# %%
%load_ext fireducks.pandas
import warnings
import os
import gc
from typing import Optional, Dict, Any, Tuple
import pickle
import time
from tqdm import tqdm

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# %%
random_state = 42
CHUNK_SIZE = 50000  # Размер чанка для обработки

# %%
class MemoryEfficientCTRPreprocessor:
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self.target_encoders = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_hasher = FeatureHasher(n_features=50, input_type="string")

        # Статистики для target encoding
        self.target_stats = {}
        self.global_mean = 0.0
        self.total_samples = 0

        # Для отслеживания уникальных значений
        self.unique_values = {}

    def reduce_memory_usage(self, df):
        """Reduce memory usage of dataframe - optimized version"""
        start_mem = df.memory_usage().sum() / 1024**2

        # Optimize only numeric columns to speed up
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        # Convert object columns to category if they have low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        if end_mem < start_mem:
            print(f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

        return df

    def engineer_features(self, df):
        """Feature engineering - optimized version"""
        # Time features - vectorized operations
        hour_str = df["hour"].astype('str')
        df["hour_parsed"] = hour_str.str[-2:].astype(np.int8)
        df["day_parsed"] = hour_str.str[-4:-2].astype(np.int8)

        # Cyclical encoding for hour - vectorized
        hour_rad = 2 * np.pi * df["hour_parsed"] / 24
        df["hour_sin"] = np.sin(hour_rad).astype(np.float32)
        df["hour_cos"] = np.cos(hour_rad).astype(np.float32)

        # Remove highly correlated features
        features_to_drop = ["C14", "C18", "hour", "id"]
        df = df.drop(columns=features_to_drop, errors="ignore")

        # Interaction features - optimized string operations
        df["banner_device_type"] = df["banner_pos"].astype(str) + "_" + df["device_type"].astype(str)
        df["app_site_match"] = (df["app_domain"].astype("string") == df["site_domain"].astype("string")).astype(np.int8)
        df = df.drop(columns=["site_domain", "app_domain"], errors="ignore")

        return df

    def collect_statistics_chunk(self, chunk_df):
        """Collect statistics from a chunk for target encoding"""
        if "click" not in chunk_df.columns:
            return

        chunk_target = chunk_df["click"]
        chunk_size = len(chunk_df)

        # Update global statistics
        self.global_mean = (self.global_mean * self.total_samples + chunk_target.sum()) / (self.total_samples + chunk_size)
        self.total_samples += chunk_size

        # Define categorical columns for target encoding
        medium_cardinality = ["site_id", "app_id", "device_model"]

        for col in medium_cardinality:
            if col in chunk_df.columns:
                if col not in self.target_stats:
                    self.target_stats[col] = {}

                # Collect statistics for each unique value
                for val in chunk_df[col].unique():
                    mask = chunk_df[col] == val
                    val_count = mask.sum()
                    val_sum = chunk_target[mask].sum()

                    if val in self.target_stats[col]:
                        self.target_stats[col][val]['count'] += val_count
                        self.target_stats[col][val]['sum'] += val_sum
                    else:
                        self.target_stats[col][val] = {'count': val_count, 'sum': val_sum}

        # Collect unique values for label encoding
        low_cardinality = [
            "site_category", "app_category", "C1", "banner_pos",
            "device_type", "device_conn_type", "banner_device_type"
        ]

        for col in low_cardinality:
            if col in chunk_df.columns:
                if col not in self.unique_values:
                    self.unique_values[col] = set()
                self.unique_values[col].update(chunk_df[col].astype(str).unique())

    def fit_encoders(self):
        """Fit encoders based on collected statistics"""
        print("Fitting encoders...")

        # Fit target encoders
        medium_cardinality = ["site_id", "app_id", "device_model"]
        for col in medium_cardinality:
            if col in self.target_stats:
                self.target_encoders[col] = {}
                for val, stats in self.target_stats[col].items():
                    # Smoothed target encoding
                    smoothing = 1.0
                    mean_target = stats['sum'] / stats['count']
                    smoothed_mean = (mean_target * stats['count'] + self.global_mean * smoothing) / (stats['count'] + smoothing)
                    self.target_encoders[col][val] = smoothed_mean

        # Fit label encoders
        low_cardinality = [
            "site_category", "app_category", "C1", "banner_pos",
            "device_type", "device_conn_type", "banner_device_type"
        ]

        for col in low_cardinality:
            if col in self.unique_values:
                le = LabelEncoder()
                le.fit(list(self.unique_values[col]))
                self.label_encoders[col] = le

    def encode_categorical_features_chunk(self, chunk_df, is_train=True):
        """Encode categorical features for a chunk - optimized version"""
        # Target encoding for medium cardinality features
        medium_cardinality = ["site_id", "app_id", "device_model"]
        for col in medium_cardinality:
            if col in chunk_df.columns and col in self.target_encoders:
                # Use map for faster lookups
                chunk_df[f"{col}_target"] = chunk_df[col].map(self.target_encoders[col])
                chunk_df[f"{col}_target"] = chunk_df[f"{col}_target"].fillna(self.global_mean).astype(np.float32)

        # Label encoding for low cardinality features
        low_cardinality = [
            "site_category", "app_category", "C1", "banner_pos",
            "device_type", "device_conn_type", "banner_device_type"
        ]

        for col in low_cardinality:
            if col in chunk_df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Faster encoding with map
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                chunk_df[f"{col}_label"] = chunk_df[col].astype(str).map(mapping).fillna(-1).astype(np.int16)

        # Hash encoding for high cardinality features
        high_cardinality = ["device_id", "device_ip"]
        for col in high_cardinality:
            if col in chunk_df.columns:
                # Device frequency encoding - optimized
                freq_map = chunk_df[col].value_counts().to_dict()
                chunk_df[f"{col}_freq"] = chunk_df[col].map(freq_map).astype(np.int16)

                # Hash encoding - batch processing
                input_data = chunk_df[col].astype(str).values.reshape(-1, 1)
                hashed_data = self.feature_hasher.transform(input_data)

                # Convert to DataFrame more efficiently
                hashed_df = pd.DataFrame(
                    hashed_data.toarray().astype(np.float32),
                    columns=[f"{col}_hash_{i}" for i in range(50)],
                    index=chunk_df.index
                )
                chunk_df = pd.concat([chunk_df, hashed_df], axis=1)

        # Drop original categorical columns
        cols_to_drop = medium_cardinality + low_cardinality + high_cardinality
        chunk_df = chunk_df.drop(columns=cols_to_drop, errors="ignore")

        return chunk_df

    def process_file_in_chunks(self, file_path: str, output_path: str, is_train: bool = True):
        """Process large file in chunks with progress tracking"""
        print(f"Processing {file_path} in chunks...")

        # Get total number of rows for progress tracking
        total_rows = sum(1 for _ in open(file_path)) - 1  # -1 for header
        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        print(f"Total rows: {total_rows:,}, Total chunks: {total_chunks}")

        # First pass: collect statistics (only for training data)
        if is_train:
            print("Pass 1/2: Collecting statistics...")
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)

            start_time = time.time()
            with tqdm(total=total_chunks, desc="Statistics") as pbar:
                for i, chunk in enumerate(chunk_iter):
                    chunk_start = time.time()

                    chunk = self.reduce_memory_usage(chunk)
                    chunk = self.engineer_features(chunk)
                    self.collect_statistics_chunk(chunk)

                    # Update progress
                    chunk_time = time.time() - chunk_start
                    pbar.set_postfix({
                        'chunk_time': f'{chunk_time:.1f}s',
                        'rows/s': f'{len(chunk)/chunk_time:.0f}',
                        'mem': f'{chunk.memory_usage().sum()/1024**2:.1f}MB'
                    })
                    pbar.update(1)

                    # Force garbage collection
                    del chunk
                    gc.collect()

            elapsed = time.time() - start_time
            print(f"Statistics collection completed in {elapsed:.1f}s ({total_rows/elapsed:.0f} rows/s)")

            # Fit encoders after collecting all statistics
            self.fit_encoders()

            # Save encoders
            self.save_encoders("encoders.pkl")
        else:
            # Load encoders for test data
            self.load_encoders("encoders.pkl")

        # Second pass: transform data
        print("Pass 2/2: Transforming data...")
        chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)

        processed_chunks = []
        numerical_stats = None

        start_time = time.time()
        with tqdm(total=total_chunks, desc="Transform") as pbar:
            for i, chunk in enumerate(chunk_iter):
                chunk_start = time.time()

                # Memory optimization
                chunk = self.reduce_memory_usage(chunk)

                # Feature engineering
                chunk = self.engineer_features(chunk)

                # Encode categorical features
                chunk = self.encode_categorical_features_chunk(chunk, is_train=is_train)

                # Separate features and target
                if is_train and "click" in chunk.columns:
                    X_chunk = chunk.drop(["click", "idx"], axis=1, errors="ignore")
                    y_chunk = chunk["click"]
                else:
                    X_chunk = chunk.drop(["idx"], axis=1, errors="ignore")
                    y_chunk = None

                # Collect numerical columns for scaling
                numerical_cols = X_chunk.select_dtypes(include=[np.number]).columns

                if is_train:
                    # Fit scaler incrementally
                    if numerical_stats is None:
                        numerical_stats = {
                            'mean': X_chunk[numerical_cols].mean(),
                            'var': X_chunk[numerical_cols].var(),
                            'count': len(X_chunk)
                        }
                    else:
                        # Update statistics incrementally
                        n1 = numerical_stats['count']
                        n2 = len(X_chunk)
                        n = n1 + n2

                        # Update mean
                        new_mean = (numerical_stats['mean'] * n1 + X_chunk[numerical_cols].mean() * n2) / n

                        # Update variance
                        new_var = ((n1 - 1) * numerical_stats['var'] + (n2 - 1) * X_chunk[numerical_cols].var() +
                                  n1 * n2 / n * (numerical_stats['mean'] - X_chunk[numerical_cols].mean()) ** 2) / (n - 1)

                        numerical_stats = {
                            'mean': new_mean,
                            'var': new_var,
                            'count': n
                        }

                # Save chunk with reset index to avoid range index issues
                X_chunk = X_chunk.reset_index(drop=True)
                if is_train:
                    y_chunk = y_chunk.reset_index(drop=True)
                    X_chunk.to_parquet(f"{output_path}_X_chunk_{i}.parquet", index=False)
                    y_chunk.to_frame().to_parquet(f"{output_path}_y_chunk_{i}.parquet", index=False)
                else:
                    X_chunk.to_parquet(f"{output_path}_X_chunk_{i}.parquet", index=False)

                processed_chunks.append(i)

                # Update progress
                chunk_time = time.time() - chunk_start
                pbar.set_postfix({
                    'chunk_time': f'{chunk_time:.1f}s',
                    'rows/s': f'{len(chunk)/chunk_time:.0f}',
                    'features': len(X_chunk.columns)
                })
                pbar.update(1)

                # Force garbage collection
                del chunk, X_chunk
                if y_chunk is not None:
                    del y_chunk
                gc.collect()

        elapsed = time.time() - start_time
        print(f"Transformation completed in {elapsed:.1f}s ({total_rows/elapsed:.0f} rows/s)")

        # Fit and save scaler
        if is_train and numerical_stats is not None:
            # Create a dummy scaler with the computed statistics
            self.scaler.mean_ = numerical_stats['mean'].values
            self.scaler.var_ = numerical_stats['var'].values
            self.scaler.scale_ = np.sqrt(numerical_stats['var'].values)
            self.scaler.n_samples_seen_ = numerical_stats['count']

            # Save scaler
            joblib.dump(self.scaler, "scaler.pkl")
        elif not is_train:
            # Load scaler for test data
            self.scaler = joblib.load("scaler.pkl")

        # Apply scaling to all chunks
        print("Pass 3/3: Applying scaling...")
        start_time = time.time()
        with tqdm(total=len(processed_chunks), desc="Scaling") as pbar:
            for i in processed_chunks:
                chunk_start = time.time()

                X_chunk = pd.read_parquet(f"{output_path}_X_chunk_{i}.parquet")

                numerical_cols = X_chunk.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    X_chunk[numerical_cols] = self.scaler.transform(X_chunk[numerical_cols])

                # Save scaled chunk with reset index
                X_chunk = X_chunk.reset_index(drop=True)
                X_chunk.to_parquet(f"{output_path}_X_chunk_{i}_scaled.parquet", index=False)

                numerical_cols = X_chunk.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    X_chunk[numerical_cols] = self.scaler.transform(X_chunk[numerical_cols])

                # Save scaled chunk
                X_chunk.to_parquet(f"{output_path}_X_chunk_{i}_scaled.parquet")

                # Update progress
                chunk_time = time.time() - chunk_start
                pbar.set_postfix({
                    'chunk_time': f'{chunk_time:.1f}s',
                    'rows': len(X_chunk)
                })
                pbar.update(1)

                # Clean up
                del X_chunk
                gc.collect()

        elapsed = time.time() - start_time
        print(f"Scaling completed in {elapsed:.1f}s")

        return processed_chunks

    def save_encoders(self, filepath: str):
        """Save all encoders to file"""
        encoders = {
            'target_encoders': self.target_encoders,
            'label_encoders': self.label_encoders,
            'feature_hasher': self.feature_hasher,
            'global_mean': self.global_mean,
            'total_samples': self.total_samples
        }

        with open(filepath, 'wb') as f:
            pickle.dump(encoders, f)

        print(f"Encoders saved to {filepath}")

    def load_encoders(self, filepath: str):
        """Load encoders from file"""
        with open(filepath, 'rb') as f:
            encoders = pickle.load(f)

        self.target_encoders = encoders['target_encoders']
        self.label_encoders = encoders['label_encoders']
        self.feature_hasher = encoders['feature_hasher']
        self.global_mean = encoders['global_mean']
        self.total_samples = encoders['total_samples']

        print(f"Encoders loaded from {filepath}")

    def load_processed_chunks(self, output_path: str, chunk_indices: list, load_target: bool = True):
        """Load processed chunks back into memory for training"""
        X_chunks = []
        y_chunks = []

        for i in chunk_indices:
            X_chunk = pd.read_parquet(f"{output_path}_X_chunk_{i}_scaled.parquet")
            X_chunks.append(X_chunk)

            if load_target:
                y_chunk = pd.read_parquet(f"{output_path}_y_chunk_{i}.parquet")
                y_chunks.append(y_chunk.iloc[:, 0])  # Get the series

        # Combine chunks
        X = pd.concat(X_chunks, ignore_index=True)
        y = pd.concat(y_chunks, ignore_index=True) if load_target else None

        print(f"Loaded {len(X)} samples with {len(X.columns)} features")

        return X, y

# Usage example
if __name__ == "__main__":
    try:
        print("Initializing memory-efficient preprocessor...")
        preprocessor = MemoryEfficientCTRPreprocessor(chunk_size=CHUNK_SIZE)

        # Process training data
        print("Processing training data...")
        train_chunks = preprocessor.process_file_in_chunks("ctr_train_sample.csv", "data/train_processed", is_train=True)

        # Process test data
        print("Processing test data...")
        test_chunks = preprocessor.process_file_in_chunks("ctr_test.csv", "data/test_processed", is_train=False)

        # Example: Load first few chunks for training
        print("Loading processed chunks for training...")
        X_train, y_train = preprocessor.load_processed_chunks("data/train_processed", train_chunks[:3])  # Load first 3 chunks

        print(f"Training data shape: {X_train.shape}")
        print(f"Class distribution: {y_train.value_counts()}")

        # Load test data
        X_test, _ = preprocessor.load_processed_chunks("data/test_processed", test_chunks, load_target=False)
        print(f"Test data shape: {X_test.shape}")

        print("Memory-efficient preprocessing completed!")
        print("Processed data:\n=== X train ===")
        print(X_train)
        print('=== y train ===')
        print(y_train)
        print('=== x test ===')
        print(X_test)

    except:
        import traceback
        print(traceback.format_exc())
