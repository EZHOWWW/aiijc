# %%
# %load_ext fireducks.pandas
import warnings

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# %%
random_state = 42

# %%
df_test = pd.read_csv("ctr_test.csv")
print(df_test)


# %%
n_sample = 10000
df_train = pd.read_csv(
    "ctr_train_part.csv"
)  # .sample(n=n_sample*10, random_state=random_state)
df_train_sample = df_train.sample(n=n_sample, random_state=random_state)
print(df_train)
# %%
df_train = df_train.drop([df_train.index[-1]])
# %%
# Basic info about datasets
print("=== TRAIN DATASET INFO ===")
print(f"Train shape: {df_train.shape}")
print("Target distribution:")
print(df_train["click"].value_counts(normalize=True))
print(f"CTR: {df_train['click'].mean():.4f}")

print("\n=== MISSING VALUES ===")
print("Train missing values:")
print(df_train.isnull().sum())
print("\nTest missing values:")
print(df_test.isnull().sum())

# Check unique values for categorical features
print("\n=== CARDINALITY ANALYSIS ===")
categorical_cols = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
]

for col in categorical_cols:
    if col in df_train.columns:
        n_unique = df_train[col].nunique()
        print(f"{col}: {n_unique:,} unique values")

# Check numerical features
print("\n=== NUMERICAL FEATURES ===")
numerical_cols = [
    "hour",
    "C1",
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

for col in numerical_cols:
    if col in df_train.columns:
        print(
            f"{col}: {df_train[col].nunique()} unique values, range: {df_train[col].min()} - {df_train[col].max()}"
        )

# Check if test idx has NaN values (seems like test set based on your output)
print("\n=== TEST SET ANALYSIS ===")
print(f"Test shape: {df_test.shape}")
print(f"Test idx NaN count: {df_test['idx'].isna().sum()}")
print(f"Test has click column: {'click' in df_test.columns}")

# Time analysis
print("\n=== TIME ANALYSIS ===")
df_train["hour_parsed"] = (
    df_train["hour"].astype(int).astype(str).str[-2:].astype(int)
)
print("Hour distribution:")
print(df_train["hour_parsed"].value_counts().sort_index())

# Check data consistency between train and test
print("\n=== TRAIN/TEST CONSISTENCY ===")
common_cols = set(df_train.columns) & set(df_test.columns)
print(f"Common columns: {len(common_cols)}")
print(f"Train-only columns: {set(df_train.columns) - common_cols}")
print(f"Test-only columns: {set(df_test.columns) - common_cols}")


# %%
# Data Visualization and EDA
plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Target distribution
ax1 = axes[0, 0]
df_train_sample["click"].value_counts().plot(
    kind="bar", ax=ax1, color=["skyblue", "orange"]
)
ax1.set_title("Target Distribution")
ax1.set_xlabel("Click")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=0)

# 2. CTR by hour
ax2 = axes[0, 1]
df_train_sample["hour_parsed"] = (
    df_train_sample["hour"].astype(int).astype(str).str[-2:].astype(int)
)
ctr_by_hour = df_train_sample.groupby("hour_parsed")["click"].agg(
    ["mean", "count"]
)
ctr_by_hour["mean"].plot(kind="bar", ax=ax2, color="green")
ax2.set_title("CTR by Hour")
ax2.set_xlabel("Hour")
ax2.set_ylabel("CTR")
ax2.tick_params(axis="x", rotation=45)

# 3. CTR by banner position
ax3 = axes[0, 2]
ctr_by_banner = df_train_sample.groupby("banner_pos")["click"].agg(
    ["mean", "count"]
)
ctr_by_banner["mean"].plot(kind="bar", ax=ax3, color="red")
ax3.set_title("CTR by Banner Position")
ax3.set_xlabel("Banner Position")
ax3.set_ylabel("CTR")

# 4. CTR by device type
ax4 = axes[1, 0]
ctr_by_device = df_train_sample.groupby("device_type")["click"].agg(
    ["mean", "count"]
)
ctr_by_device["mean"].plot(kind="bar", ax=ax4, color="purple")
ax4.set_title("CTR by Device Type")
ax4.set_xlabel("Device Type")
ax4.set_ylabel("CTR")

# 5. CTR by device connection type
ax5 = axes[1, 1]
ctr_by_conn = df_train_sample.groupby("device_conn_type")["click"].agg(
    ["mean", "count"]
)
ctr_by_conn["mean"].plot(kind="bar", ax=ax5, color="brown")
ax5.set_title("CTR by Device Connection Type")
ax5.set_xlabel("Device Connection Type")
ax5.set_ylabel("CTR")

# 6. Distribution of C14 (seems to be important numerical feature)
ax6 = axes[1, 2]
df_train_sample["C14"].hist(bins=50, ax=ax6, alpha=0.7, color="cyan")
ax6.set_title("Distribution of C14")
ax6.set_xlabel("C14")
ax6.set_ylabel("Frequency")

plt.tight_layout()
# plt.show()

# Correlation analysis for numerical features
print("\n=== CORRELATION ANALYSIS ===")
numerical_features = [
    "hour_parsed",
    "C1",
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "click",
]

# Sample data for correlation (too big dataset)

correlation_matrix = df_train_sample[numerical_features].corr()
print(correlation_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    fmt=".3f",
)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
# plt.show()

# Feature importance analysis - check which features have different distributions for click vs no-click
print("\n=== FEATURE IMPORTANCE PREVIEW ===")
for col in ["banner_pos", "device_type", "device_conn_type", "C1"]:
    print(f"\n{col}:")
    cross_tab = pd.crosstab(
        df_train_sample[col], df_train_sample["click"], normalize="index"
    )
    print(cross_tab.round(4))
# %%
# Categorical Features Deep Analysis
print("=== HIGH CARDINALITY FEATURES ANALYSIS ===")

# Define categorical columns
categorical_cols = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
]

# Sample for analysis
sample_size = 50000  # Larger sample for categorical analysis
df_sample = df_train.sample(n=sample_size, random_state=random_state)

# Analyze each categorical feature
for col in categorical_cols:
    if col in df_sample.columns:
        print(f"\n=== {col.upper()} ===")

        # Basic stats
        n_unique = df_sample[col].nunique()
        total_unique = df_train[col].nunique()
        print(f"Unique values in sample: {n_unique:,}")
        print(f"Total unique values: {total_unique:,}")

        # CTR analysis for top categories
        ctr_by_category = df_sample.groupby(col)["click"].agg(
            ["mean", "count", "sum"]
        )
        ctr_by_category = ctr_by_category.sort_values("count", ascending=False)

        print("Top 10 categories by volume:")
        print(ctr_by_category.head(10))

        # CTR variance analysis
        ctr_stats = ctr_by_category["mean"].describe()
        print(
            f"CTR statistics: min={ctr_stats['min']:.4f}, max={ctr_stats['max']:.4f}, std={ctr_stats['std']:.4f}"
        )

        # Check for categories with extreme CTR (potential for feature engineering)
        high_ctr = ctr_by_category[ctr_by_category["mean"] > 0.3]
        low_ctr = ctr_by_category[ctr_by_category["mean"] < 0.05]

        print(f"High CTR categories (>30%): {len(high_ctr)}")
        print(f"Low CTR categories (<5%): {len(low_ctr)}")

        # Check frequency distribution
        freq_distribution = ctr_by_category["count"].describe()
        print(f"Frequency distribution: {freq_distribution}")

        print("-" * 50)

# Check for potential data leakage or time-based patterns
print("\n=== TEMPORAL PATTERNS ===")
df_sample["hour_parsed"] = (
    df_sample["hour"].astype(int).astype(str).str[-2:].astype(int)
)
df_sample["day"] = (
    df_sample["hour"].astype(int).astype(str).str[-4:-2].astype(int)
)

print("CTR by day:")
ctr_by_day = df_sample.groupby("day")["click"].agg(["mean", "count"])
print(ctr_by_day)

print("\nCTR by hour of day:")
ctr_by_hour = df_sample.groupby("hour_parsed")["click"].agg(["mean", "count"])
print(ctr_by_hour)

# Check if there are any patterns in ID features that might indicate time-based splitting
print("\n=== ID PATTERNS ===")
print("ID statistics:")
print(df_sample["id"].describe())

# Check for potential issues with test set
print("\n=== TRAIN/TEST OVERLAP CHECK ===")
# This is important for preventing data leakage
print("Checking if we need to be careful about temporal splits...")
print(f"Train hour range: {df_train['hour'].min()} - {df_train['hour'].max()}")
print(f"Test hour range: {df_test['hour'].min()} - {df_test['hour'].max()}")
# %%


class CTRPreprocessor:
    def __init__(self):
        self.target_encoders = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_hasher = FeatureHasher(n_features=50, input_type="string")

    def engineer_features(self, df):
        """Feature engineering based on EDA insights"""
        df = df.copy()

        # Time features
        df["hour_parsed"] = (
            df["hour"].astype(int).astype(str).str[-2:].astype(int)
        )
        df["day_parsed"] = (
            df["hour"].astype(int).astype(str).str[-4:-2].astype(int)
        )

        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_parsed"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_parsed"] / 24)

        # Remove highly correlated features (keep the one with higher target correlation)
        # From correlation analysis: C14 and C17 are highly correlated (0.97)
        # C18 and C21 are negatively correlated (-0.54)
        # Keep C17 (slightly higher correlation with target), remove C14
        # Keep C21 (higher correlation with target), remove C18

        features_to_drop = [
            "C14",
            "C18",
            "hour",
            "id",
        ]  # Remove original hour and id
        df = df.drop(columns=features_to_drop, errors="ignore")

        # Interaction features for important combinations
        df["banner_device_type"] = (
            df["banner_pos"].astype(str) + "_" + df["device_type"].astype(str)
        )
        df["app_site_match"] = (df["app_domain"] == df["site_domain"]).astype(
            int
        )
        df = df.drop(columns=["site_domain", "app_domain"], errors="ignore")

        # Device frequency encoding (for device_ip and device_id)
        if "device_ip" in df.columns:
            df["device_ip_freq"] = df.groupby("device_ip")[
                "device_ip"
            ].transform("count")
        if "device_id" in df.columns:
            df["device_id_freq"] = df.groupby("device_id")[
                "device_id"
            ].transform("count")

        return df

    def encode_categorical_features(
        self, df_train, df_test=None, target_col="click"
    ):
        """Encode categorical features using different strategies based on cardinality"""

        # Define categorical columns by cardinality
        low_cardinality = [
            "site_category",
            "app_category",
            "C1",
            "banner_pos",
            "device_type",
            "device_conn_type",
            "banner_device_type",
        ]

        medium_cardinality = ["site_id", "app_id", "device_model"]

        high_cardinality = ["device_id", "device_ip"]

        # For training, we have target

        if target_col in df_train.columns:
            y_train = df_train[target_col]

            # Target encoding for medium cardinality features
            print("=== medium ===")
            for col in medium_cardinality:
                if col in df_train.columns:
                    te = TargetEncoder(
                        cols=[col],
                        smoothing=1.0,
                    )
                    df_train[f"{col}_target"] = te.fit_transform(
                        df_train[col], y_train
                    )
                    self.target_encoders[col] = te

                    if df_test is not None:
                        train_cats = set(df_train[col].unique())
                        df_test[f"{col}_target"] = te.transform(
                            df_test[col][df_test[col].isin(train_cats)]
                        )
                        df_test[f"{col}_target"] = df_test[
                            f"{col}_target"
                        ].fillna(y_train.mean())

            # Label encoding for low cardinality features
            print("=== low ===")
            for col in low_cardinality:
                if col in df_train.columns:
                    le = LabelEncoder()
                    df_train[f"{col}_label"] = le.fit_transform(
                        df_train[col].astype(str)
                    )
                    self.label_encoders[col] = le

                    if df_test is not None:
                        # Handle unseen categories
                        test_values = df_test[col].astype(str)
                        test_encoded = []
                        for val in test_values:
                            if val in le.classes_:
                                test_encoded.append(le.transform([val])[0])
                            else:
                                test_encoded.append(-1)  # Unknown category
                        df_test[f"{col}_label"] = test_encoded

            print("=== high ==")
            # Hash encoding for high cardinality features
            for col in high_cardinality:
                if col in df_train.columns:
                    # Преобразуем в список списков со строками
                    train_input = [[str(val)] for val in df_train[col]]
                    train_hashed = self.feature_hasher.transform(train_input)
                    train_hashed_df = pd.DataFrame(
                        train_hashed.toarray(),
                        columns=[f"{col}_hash_{i}" for i in range(50)],
                    )
                    df_train = pd.concat(
                        [df_train.reset_index(drop=True), train_hashed_df],
                        axis=1,
                    )

                    if df_test is not None:
                        test_input = [[str(val)] for val in df_test[col]]
                        test_hashed = self.feature_hasher.transform(test_input)
                        test_hashed_df = pd.DataFrame(
                            test_hashed.toarray(),
                            columns=[f"{col}_hash_{i}" for i in range(50)],
                        )
                        df_test = pd.concat(
                            [df_test.reset_index(drop=True), test_hashed_df],
                            axis=1,
                        )

        # Drop original categorical columns
        cols_to_drop = low_cardinality + medium_cardinality + high_cardinality
        df_train = df_train.drop(columns=cols_to_drop, errors="ignore")
        if df_test is not None:
            df_test = df_test.drop(columns=cols_to_drop, errors="ignore")

        return df_train, df_test

    def prepare_data(self, df_train, df_test=None):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing...")

        # Feature engineering
        print("Engineering features...")
        df_train_proc = self.engineer_features(df_train)
        df_test_proc = (
            self.engineer_features(df_test) if df_test is not None else None
        )

        # Encode categorical features
        print("Encoding categorical features...")
        df_train_proc, df_test_proc = self.encode_categorical_features(
            df_train_proc, df_test_proc
        )

        # Separate features and target
        if "click" in df_train_proc.columns:
            X_train = df_train_proc.drop(
                ["click", "idx"], axis=1, errors="ignore"
            )
            y_train = df_train_proc["click"]
        else:
            X_train = df_train_proc.drop(["idx"], axis=1, errors="ignore")
            y_train = None

        if df_test_proc is not None:
            X_test = df_test_proc.drop(["idx"], axis=1, errors="ignore")
        else:
            X_test = None

        # Scale numerical features
        print("Scaling features...")
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        print("site\n", X_train["site_id_target"])
        X_train[numerical_cols] = self.scaler.fit_transform(
            X_train[numerical_cols]
        )

        if X_test is not None:
            X_test[numerical_cols] = self.scaler.transform(
                X_test[numerical_cols]
            )

        print(f"Final feature shape: {X_train.shape}")
        print(f"Features: {list(X_train.columns)}")

        return X_train, X_test, y_train


# Memory optimization function
def reduce_memory_usage(df):
    """Reduce memory usage of dataframe"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


# Usage example
print("Initializing preprocessor...")
preprocessor = CTRPreprocessor()

# Memory optimization
print("Optimizing memory usage...")
df_train = reduce_memory_usage(df_train)
df_test = reduce_memory_usage(df_test)

# %%
# Prepare data


import traceback

try:
    X_train, X_test, y_train = preprocessor.prepare_data(df_train, df_test)
except Exception:
    print(traceback.format_exc())

print(y_train)
# Check class distribution
print("\nClass distribution:")
print(f"Class 0: {(y_train == 0).sum():,} ({(y_train == 0).mean():.3f})")
print(f"Class 1: {(y_train == 1).sum():,} ({(y_train == 1).mean():.3f})")

# Save processed data
print("Saving processed data...")
X_train.to_parquet("X_train_processed.parquet")
X_test.to_parquet("X_test_processed.parquet")
y_train.to_frame().to_parquet("y_train_processed.parquet")

print("Preprocessing completed!")


# %%
class CTRModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}

    def create_models(self):
        """Create different models for comparison"""
        models = {
            "sgd": SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                random_state=42,
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
                random_state=42,
                n_estimators=1000,
                early_stopping_rounds=100,
                class_weight="balanced",
            ),
            "catboost": cb.CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                random_seed=42,
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

    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        balance_method="smote",
        use_cv=True,
    ):
        """Train and evaluate models"""

        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(
            X_train, y_train, balance_method
        )

        models = self.create_models()
        results = {}

        for name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"Training {name.upper()} model...")
            print(f"{'=' * 50}")

            try:
                # Train model
                if name == "lgb":
                    model.fit(
                        X_train_balanced,
                        y_train_balanced,
                        eval_set=[(X_val, y_val)],
                        eval_metric="auc",
                        callbacks=[
                            lgb.early_stopping(100),
                            lgb.log_evaluation(0),
                        ],
                    )
                elif name == "catboost":
                    X_train_balanced_pd = (
                        X_train_balanced.to_pandas()
                        if hasattr(X_train_balanced, "to_pandas")
                        else X_train_balanced
                    )
                    y_train_balanced_pd = (
                        y_train_balanced.to_pandas()
                        if hasattr(y_train_balanced, "to_pandas")
                        else y_train_balanced
                    )
                    X_val_pd = (
                        X_val.to_pandas()
                        if hasattr(X_val, "to_pandas")
                        else X_val
                    )
                    y_val_pd = (
                        y_val.to_pandas()
                        if hasattr(y_val, "to_pandas")
                        else y_val
                    )
                    # X_train_balanced_pd = pandas.DataFrame(X_train_balanced_pd)
                    # y_train_balanced_pd = pandas.Series(y_train_balanced_pd)
                    # X_val_pd = pandas.DataFrame(X_val_pd)
                    # y_val_pd = pandas.Series(y_val_pd)

                    print(
                        type(X_train_balanced_pd)
                    )  # должно быть <class 'pandas.core.frame.DataFrame'>
                    print(isinstance(X_train_balanced_pd, pd.DataFrame))  # True
                    model.fit(
                        X_train_balanced_pd,
                        y_train_balanced_pd,
                        eval_set=(X_val_pd, y_val_pd),
                        verbose=False,
                    )
                else:
                    model.fit(X_train_balanced, y_train_balanced)

                # Predictions
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # Evaluate
                auc_score = roc_auc_score(y_val, y_pred_proba)

                # Cross-validation on original data (time-based)
                if use_cv and name != "catboost":  # CatBoost CV is different
                    cv_scores = []
                    skf = StratifiedKFold(
                        n_splits=5, shuffle=True, random_state=42
                    )

                    for train_idx, val_idx in skf.split(X_train, y_train):
                        X_cv_train = X_train.iloc[train_idx]
                        X_cv_val = X_train.iloc[val_idx]
                        y_cv_train = y_train.iloc[train_idx]
                        y_cv_val = y_train.iloc[val_idx]

                        # Balance CV training data
                        X_cv_balanced, y_cv_balanced = self.handle_imbalance(
                            X_cv_train, y_cv_train, balance_method
                        )

                        # Train and predict
                        if name == "lgb":
                            model_cv = lgb.LGBMClassifier(**model.get_params())
                            model_cv.fit(
                                X_cv_balanced,
                                y_cv_balanced,
                                eval_set=[(X_cv_val, y_cv_val)],
                                eval_metric="auc",
                                callbacks=[
                                    lgb.early_stopping(50),
                                    lgb.log_evaluation(0),
                                ],
                            )
                        else:
                            model_cv = type(model)(**model.get_params())
                            model_cv.fit(X_cv_balanced, y_cv_balanced)

                        y_cv_pred = model_cv.predict_proba(X_cv_val)[:, 1]
                        cv_scores.append(roc_auc_score(y_cv_val, y_cv_pred))

                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    print(f"CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

                    results[name] = {
                        "model": model,
                        "val_auc": auc_score,
                        "cv_auc_mean": cv_mean,
                        "cv_auc_std": cv_std,
                        "cv_scores": cv_scores,
                    }
                else:
                    results[name] = {
                        "model": model,
                        "val_auc": auc_score,
                        "cv_auc_mean": auc_score,
                        "cv_auc_std": 0.0,
                        "cv_scores": [auc_score],
                    }

                print(f"Validation AUC: {auc_score:.4f}")

                # Feature importance (if available)
                if hasattr(model, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X_train.columns,
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

        # Find best model
        best_model_name = max(
            results.keys(), key=lambda x: results[x]["val_auc"]
        )
        best_model = results[best_model_name]["model"]

        print(f"\n{'=' * 50}")
        print(f"BEST MODEL: {best_model_name.upper()}")
        print(f"Validation AUC: {results[best_model_name]['val_auc']:.4f}")
        print(
            f"CV AUC: {results[best_model_name]['cv_auc_mean']:.4f} (+/- {results[best_model_name]['cv_auc_std']:.4f})"
        )
        print(f"{'=' * 50}")

        return results, best_model, best_model_name

    def make_predictions(self, model, X_test):
        """Make predictions on test set"""
        predictions = model.predict_proba(X_test)[:, 1]
        return predictions

    def save_submission(self, predictions, filename="submission.csv"):
        """Save predictions in submission format"""
        submission = pd.DataFrame(
            {"idx": range(len(predictions)), "click": predictions}
        )
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        return submission


# Initialize trainer
trainer = CTRModelTrainer()

# Create time-based split
print("Creating time-based validation split...")
X_train_split, X_val_split, y_train_split, y_val_split = (
    trainer.time_based_split(X_train, y_train, test_size=0.2)
)

print(f"Training set: {X_train_split.shape}")
print(f"Validation set: {X_val_split.shape}")


# %%
# Train models
print("Training models...")
results, best_model, best_model_name = trainer.train_and_evaluate(
    X_train_split,
    y_train_split,
    X_val_split,
    y_val_split,
    balance_method="undersample",
    use_cv=True,
)

# %%
# Make predictions on test set
print("Making predictions on test set...")
test_predictions = trainer.make_predictions(best_model, X_test)

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
