# %%
import gc
import glob
import warnings

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
# %%
random_state = 43


# %%
class CTRModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.preprocessor = None

    def load_preprocessor(self, preprocessor_path="preprocessed.py"):
        """Load the preprocessor class"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "preprocessor", preprocessor_path
        )
        preprocessor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocessor_module)

        self.preprocessor = preprocessor_module.MemoryEfficientCTRPreprocessor(
            chunk_size=100000
        )

    def get_chunk_files(self, data_path, data_type="train"):
        """Get list of chunk files"""
        if data_type == "train":
            X_files = sorted(glob.glob(f"{data_path}_X_chunk_*_scaled.parquet"))
            y_files = sorted(glob.glob(f"{data_path}_y_chunk_*.parquet"))
            return X_files, y_files
        else:
            X_files = sorted(glob.glob(f"{data_path}_X_chunk_*_scaled.parquet"))
            return X_files, None

    def load_chunk_batch(
        self, X_files, y_files=None, start_idx=0, batch_size=3
    ):
        """Load a batch of chunks"""
        end_idx = min(start_idx + batch_size, len(X_files))

        X_chunks = []
        y_chunks = []

        for i in range(start_idx, end_idx):
            # Load X chunk
            X_chunk = pd.read_parquet(X_files[i])
            X_chunks.append(X_chunk)

            # Load y chunk if training data
            if y_files is not None:
                y_chunk = pd.read_parquet(y_files[i])
                y_chunks.append(y_chunk.iloc[:, 0])  # Get the series

        # Combine chunks
        X = pd.concat(X_chunks, ignore_index=True)
        y = (
            pd.concat(y_chunks, ignore_index=True)
            if y_files is not None
            else None
        )

        print(f"Loaded batch: {len(X)} samples with {len(X.columns)} features")

        return X, y

    def create_models(self):
        """Create different models for comparison"""
        models = {
            "sgd": SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                random_state=random_state,
                max_iter=1000,
                class_weight="balanced",
            ),
            "lgb": lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=random_state,
                n_estimators=1000,
                early_stopping_rounds=100,
                class_weight="balanced",
            ),
            "catboost": cb.CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                random_seed=random_state,
                verbose=False,
                early_stopping_rounds=100,
            ),
        }
        return models

    def time_based_split(self, X, y, test_size=0.2):
        """Time-based split for CTR data"""
        # Assuming data is already sorted by time
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]

        return X_train, X_val, y_train, y_val

    def handle_imbalance(self, X_train, y_train, method="smote"):
        """Handle class imbalance"""
        print(
            f"Original class distribution: {y_train.value_counts().to_dict()}"
        )

        # X_train = X_train.drop(columns=["app_site_match"])
        if method == "smote":
            # SMOTE for oversampling
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        elif method == "undersample":
            # Random undersampling (faster for large datasets)
            rus = RandomUnderSampler(
                random_state=random_state, sampling_strategy=0.5
            )  # 1:2 ratio
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

        elif method == "combined":
            # Combined approach: oversample minority, undersample majority
            # First oversample minority class
            smote = SMOTE(random_state=random_state, sampling_strategy=0.3)
            X_temp, y_temp = smote.fit_resample(X_train, y_train)

            # Then undersample majority class
            rus = RandomUnderSampler(
                random_state=random_state, sampling_strategy=0.7
            )
            X_resampled, y_resampled = rus.fit_resample(X_temp, y_temp)

        else:
            X_resampled, y_resampled = X_train, y_train

        print(
            f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}"
        )

        return X_resampled, y_resampled

    def train_incremental_sgd(self, X_files, y_files, validation_data=None):
        """Train SGD model incrementally on batches"""
        print("Training SGD model incrementally...")

        # Initialize SGD model
        sgd_model = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            random_state=random_state,
            max_iter=1,  # One iteration per batch
            class_weight="balanced",
        )

        batch_size = 2  # Number of chunks per batch
        n_batches = len(X_files) // batch_size + (
            1 if len(X_files) % batch_size != 0 else 0
        )

        print(f"Training on {n_batches} batches...")

        # Initialize model with first batch
        X_batch, y_batch = self.load_chunk_batch(
            X_files, y_files, 0, batch_size
        )

        # Handle imbalance for first batch
        X_batch_balanced, y_batch_balanced = self.handle_imbalance(
            X_batch, y_batch
        )

        # Fit initial model
        sgd_model.fit(X_batch_balanced, y_batch_balanced)

        del X_batch, y_batch, X_batch_balanced, y_batch_balanced
        gc.collect()

        # Incremental training on remaining batches
        for batch_idx in tqdm(range(1, n_batches), desc="Training SGD"):
            start_idx = batch_idx * batch_size

            X_batch, y_batch = self.load_chunk_batch(
                X_files, y_files, start_idx, batch_size
            )

            # Handle imbalance
            if X_batch.shape[0] != y_batch.shape[0]:
                print(
                    f"=== ERROR: Shape mismatch! X_batch has {X_batch.shape[0]} rows, but y_batch has {y_batch.shape[0]} rows. Skipping batch."
                )
                continue
            X_batch_balanced, y_batch_balanced = self.handle_imbalance(
                X_batch, y_batch
            )

            # Partial fit
            sgd_model.partial_fit(X_batch_balanced, y_batch_balanced)

            del X_batch, y_batch, X_batch_balanced, y_batch_balanced
            gc.collect()

        # Evaluate on validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_pred_proba = sgd_model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred_proba)
            print(f"SGD Validation AUC: {auc_score:.4f}")
            return sgd_model, auc_score

        return sgd_model, None

    def train_batch_models(
        self, X_files, y_files, validation_data=None, batch_size=3
    ):
        """Train models on batches of data"""
        print(
            f"Training models on {len(X_files)} chunks in batches of {batch_size}..."
        )

        # Load first batch for training
        X_batch, y_batch = self.load_chunk_batch(
            X_files, y_files, 0, batch_size
        )

        # Create validation split from first batch if no validation data provided
        if validation_data is None:
            X_train_split, X_val_split, y_train_split, y_val_split = (
                self.time_based_split(X_batch, y_batch, test_size=0.2)
            )
        else:
            X_train_split, y_train_split = X_batch, y_batch
            X_val_split, y_val_split = validation_data

        print(f"Training set: {X_train_split.shape}")
        print(f"Validation set: {X_val_split.shape}")

        # X_batch = X_batch.drop(columns=["app_site_match"])
        # X_val_split = X_val_split.drop(columns=["app_site_match"])
        # X_train_split = X_train_split.drop(columns=["app_site_match"])

        # Train models
        results = {}
        models = self.create_models()

        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"Training {name.upper()} model...")
            print(f"{'=' * 50}")

            try:
                # Handle class imbalance
                X_train_balanced, y_train_balanced = self.handle_imbalance(
                    X_train_split, y_train_split
                )

                # Train model
                if name == "lgb":
                    model.fit(
                        X_train_balanced,
                        y_train_balanced,
                        eval_set=[(X_val_split, y_val_split)],
                        eval_metric="auc",
                        callbacks=[
                            lgb.early_stopping(100),
                            lgb.log_evaluation(0),
                        ],
                    )
                elif name == "catboost":
                    model.fit(
                        X_train_balanced,
                        y_train_balanced,
                        eval_set=(X_val_split, y_val_split),
                        verbose=False,
                    )
                else:
                    model.fit(X_train_balanced, y_train_balanced)

                # Predictions
                y_pred_proba = model.predict_proba(X_val_split)[:, 1]

                # Evaluate
                auc_score = roc_auc_score(y_val_split, y_pred_proba)

                results[name] = {
                    "model": model,
                    "val_auc": auc_score,
                }

                print(f"Validation AUC: {auc_score:.4f}")

                # Feature importance (if available)
                if hasattr(model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X_train_split.columns,
                            "importance": model.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)

                    print(f"\nTop 10 features for {name}:")
                    print(feature_importance.head(10))

            except Exception as e:
                import traceback

                print(f"Error training {name}: {str(e)}")
                print(traceback.format_exc())
                continue

        # Train incremental SGD on all batches
        print("\n" + "=" * 50)
        print("Training incremental SGD on all data...")
        print("=" * 50)

        sgd_incremental, sgd_auc = self.train_incremental_sgd(
            X_files, y_files, (X_val_split, y_val_split)
        )

        if sgd_auc is not None:
            results["sgd_incremental"] = {
                "model": sgd_incremental,
                "val_auc": sgd_auc,
            }

        # Find best model
        if results:
            best_model_name = max(
                results.keys(), key=lambda x: results[x]["val_auc"]
            )
            best_model = results[best_model_name]["model"]

            print(f"\n{'=' * 50}")
            print(f"BEST MODEL: {best_model_name.upper()}")
            print(f"Validation AUC: {results[best_model_name]['val_auc']:.4f}")
            print(f"{'=' * 50}")

            return results, best_model, best_model_name

        return results, None, None

    def make_predictions_on_chunks(self, model, X_test_files, batch_size=3):
        """Make predictions on test chunks"""
        print("Making predictions on test data...")

        all_predictions = []
        n_batches = len(X_test_files) // batch_size + (
            1 if len(X_test_files) % batch_size != 0 else 0
        )

        for batch_idx in tqdm(range(n_batches), desc="Predicting"):
            start_idx = batch_idx * batch_size
            X_batch, _ = self.load_chunk_batch(
                X_test_files, None, start_idx, batch_size
            )

            # Make predictions
            batch_predictions = model.predict_proba(X_batch)[:, 1]
            all_predictions.extend(batch_predictions)

            del X_batch
            gc.collect()

        return np.array(all_predictions)

    def save_submission(self, predictions, filename="submission.csv"):
        """Save predictions in submission format"""
        example_submission = pd.read_csv('ctr_sample_submission.csv')
        print(example_submission)
        submission = pd.DataFrame(
            {"idx": example_submission['idx'], "click": predictions}
        )
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        return submission


# %%
# Initialize trainer
trainer = CTRModelTrainer()

# Get chunk files
print("Loading chunk files...")
X_train_files, y_train_files = trainer.get_chunk_files(
    "data/train_processed", "train"
)
X_train_files = X_train_files[:20]
X_test_files, _ = trainer.get_chunk_files("data/test_processed", "test")

print(f"Found {len(X_train_files)} training chunk files")
print(f"Found {len(X_test_files)} test chunk files")

if len(X_train_files) == 0:
    print(
        "No training chunk files found! Make sure preprocessed.py has been run."
    )
print(X_test_files)
# %%
# Train models
print("Training models...")
try:
    results, best_model, best_model_name = trainer.train_batch_models(
        X_train_files, y_train_files, batch_size=3
    )
except:
    import traceback

    print(traceback.format_exc())

if best_model is None:
    print("No models were trained successfully!")

# %%
# Make predictions on test set
print("Making predictions on test set...")
test_predictions = trainer.make_predictions_on_chunks(
    best_model, X_test_files, batch_size=3
)
print(test_predictions)

# Save submission
submission = trainer.save_submission(test_predictions, "ctr_submission.csv")

print("\nPrediction statistics:")
print(f"Min: {test_predictions.min():.4f}")
print(f"Max: {test_predictions.max():.4f}")
print(f"Mean: {test_predictions.mean():.4f}")
print(f"Std: {test_predictions.std():.4f}")

# Save best model
joblib.dump(best_model, f"best_model_{best_model_name}.pkl")
print(f"Best model saved as 'best_model_{best_model_name}.pkl'")
