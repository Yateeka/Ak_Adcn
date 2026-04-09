"""
Preprocess CIC-IDS2017 into Harmony-style vector search inputs.

Steps:
1. Read all CSV files from data/raw/cicids2017
2. Merge and clean the dataset
3. Keep numeric flow features only
4. Remove identifier and metadata columns
5. Standardize features
6. Reduce dimensionality with PCA
7. L2-normalize vectors
8. Split into base/query sets
9. Build uniform and skewed query workloads
10. Save outputs under data/processed
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize


RAW_DIR = Path("data/raw/cicids2017")
OUT_DIR = Path("data/processed")
RANDOM_STATE = 42
PCA_DIMS = 64
QUERY_FRACTION = 0.10
QUERY_SAMPLE_SIZE = 5000
SKEW_RATIO = 0.80


def find_csv_files(raw_dir: Path) -> list[Path]:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return csv_files


def load_and_merge(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged.columns = [c.strip() for c in merged.columns]
    return merged


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    label_col = "Label"
    if label_col not in df.columns:
        raise KeyError("Expected a 'Label' column in CIC-IDS2017 data")

    labels = df[label_col].astype(str).str.strip()

    drop_cols = [
        "Label",
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP",
        "Timestamp",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]

    x = df.drop(columns=existing_drop, errors="ignore")
    x = x.select_dtypes(include=[np.number])
    x = x.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~x.isna().any(axis=1)
    x = x.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].reset_index(drop=True)

    return x, labels


def vectorize_features(x: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_components = min(PCA_DIMS, x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)

    x_vec = normalize(x_pca, norm="l2")
    return x_vec


def split_base_query(x_vec: np.ndarray, labels: pd.Series):
    x_base, x_query, y_base, y_query = train_test_split(
        x_vec,
        labels,
        test_size=QUERY_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return x_base, x_query, y_base.reset_index(drop=True), y_query.reset_index(drop=True)


def build_uniform_queries(x_query: np.ndarray, size: int) -> np.ndarray:
    n = min(size, len(x_query))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(x_query), size=n, replace=False)
    return x_query[idx]


def build_skewed_queries(x_query: np.ndarray, y_query: pd.Series, size: int) -> np.ndarray:
    hot_mask = (
        y_query.str.contains("DDoS", case=False, na=False)
        | y_query.str.contains("DoS", case=False, na=False)
    )

    x_hot = x_query[hot_mask.to_numpy()]
    x_cold = x_query[~hot_mask.to_numpy()]

    if len(x_hot) == 0 or len(x_cold) == 0:
        raise ValueError("Could not build skewed queries because hot or cold group is empty")

    n_total = min(size, len(x_query))
    n_hot = min(int(n_total * SKEW_RATIO), len(x_hot))
    n_cold = min(n_total - n_hot, len(x_cold))

    rng = np.random.default_rng(RANDOM_STATE)
    hot_idx = rng.choice(len(x_hot), size=n_hot, replace=False)
    cold_idx = rng.choice(len(x_cold), size=n_cold, replace=False)

    skewed = np.vstack([x_hot[hot_idx], x_cold[cold_idx]])
    rng.shuffle(skewed)
    return skewed


def save_outputs(
    merged_df: pd.DataFrame,
    x_base: np.ndarray,
    x_query: np.ndarray,
    y_base: pd.Series,
    y_query: pd.Series,
    queries_uniform: np.ndarray,
    queries_skewed: np.ndarray,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(OUT_DIR / "merged_clean.parquet", index=False)
    np.save(OUT_DIR / "base_vectors.npy", x_base)
    np.save(OUT_DIR / "query_vectors.npy", x_query)
    np.save(OUT_DIR / "queries_uniform.npy", queries_uniform)
    np.save(OUT_DIR / "queries_skewed.npy", queries_skewed)

    pd.DataFrame({"label": y_base}).to_csv(OUT_DIR / "base_labels.csv", index=False)
    pd.DataFrame({"label": y_query}).to_csv(OUT_DIR / "query_labels.csv", index=False)


def main() -> None:
    csv_files = find_csv_files(RAW_DIR)
    merged = load_and_merge(csv_files)
    x, labels = clean_dataframe(merged)
    x_vec = vectorize_features(x)
    x_base, x_query, y_base, y_query = split_base_query(x_vec, labels)
    queries_uniform = build_uniform_queries(x_query, QUERY_SAMPLE_SIZE)
    queries_skewed = build_skewed_queries(x_query, y_query, QUERY_SAMPLE_SIZE)
    clean_merged = x.copy()
    clean_merged["Label"] = labels
    save_outputs(
        clean_merged,
        x_base,
        x_query,
        y_base,
        y_query,
        queries_uniform,
        queries_skewed,
    )

    print("Done")
    print(f"CSV files found: {len(csv_files)}")
    print(f"Clean rows: {len(clean_merged)}")
    print(f"Feature dimensions after PCA: {x_base.shape[1]}")
    print(f"Base vectors shape: {x_base.shape}")
    print(f"Query vectors shape: {x_query.shape}")
    print(f"Uniform queries shape: {queries_uniform.shape}")
    print(f"Skewed queries shape: {queries_skewed.shape}")


if __name__ == "__main__":
    main()r
