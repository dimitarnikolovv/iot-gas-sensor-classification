"""Train models for the IoT gas sensor classification project.

This script is optional. The main project is in the Jupyter notebook.

Run it with:
    python src/train.py
"""

from pathlib import Path
import zipfile
import urllib.request

import pandas as pd
from fastai.tabular.all import (
    CategoryBlock,
    ClassificationInterpretation,
    Normalize,
    RandomSplitter,
    TabularPandas,
    accuracy,
    range_of,
    tabular_learner,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ZIP_PATH = DATA_DIR / "gas_sensor_array_drift_dataset.zip"

DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
    "https://archive.ics.uci.edu/static/public/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip",
]

GAS_LABELS = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}


def download_dataset_if_needed() -> None:
    """Download the UCI dataset ZIP file if it does not already exist."""
    if ZIP_PATH.exists():
        print(f"Dataset already exists: {ZIP_PATH}")
        return

    last_error = None

    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset from: {url}")
            urllib.request.urlretrieve(url, ZIP_PATH)
            print(f"Saved to: {ZIP_PATH}")
            return
        except Exception as error:
            last_error = error
            print("Download failed for this URL. Trying next one...")

    raise RuntimeError(
        "Dataset download failed. Open the UCI page manually, download the ZIP file, "
        f"and place it here: {ZIP_PATH}"
    ) from last_error


def parse_libsvm_line(line: str, batch_name: str) -> dict:
    """Parse one row from the .dat files."""
    parts = line.strip().split()

    target_id = int(parts[0])
    row = {
        "target_id": target_id,
        "target": GAS_LABELS[target_id],
        "batch": batch_name,
    }

    for item in parts[1:]:
        feature_id, value = item.split(":")
        row[f"feature_{int(feature_id):03d}"] = float(value)

    return row


def load_dataset() -> pd.DataFrame:
    """Load and parse the UCI Gas Sensor Array Drift dataset."""
    download_dataset_if_needed()

    rows = []

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        batch_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".dat") and "batch" in Path(name).name.lower()
        ]

        batch_files = sorted(
            batch_files,
            key=lambda name: int("".join(ch for ch in Path(name).stem if ch.isdigit())),
        )

        for file_name in batch_files:
            batch_name = Path(file_name).stem

            with archive.open(file_name) as file:
                for raw_line in file:
                    line = raw_line.decode("utf-8").strip()

                    if line:
                        rows.append(parse_libsvm_line(line, batch_name))

    df = pd.DataFrame(rows)

    feature_columns = sorted([column for column in df.columns if column.startswith("feature_")])
    df = df[["target_id", "target", "batch"] + feature_columns]

    return df


def train_fastai_model(df: pd.DataFrame):
    """Train a FastAI tabular model."""
    cont_names = [column for column in df.columns if column.startswith("feature_")]

    splitter = RandomSplitter(valid_pct=0.2, seed=42)
    splits = splitter(range_of(df))

    tabular_data = TabularPandas(
        df,
        procs=[Normalize],
        cat_names=[],
        cont_names=cont_names,
        y_names="target",
        y_block=CategoryBlock,
        splits=splits,
    )

    dls = tabular_data.dataloaders(bs=64)

    learn = tabular_learner(
        dls,
        layers=[200, 100],
        metrics=accuracy,
    )

    learn.fit_one_cycle(10, 1e-3)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(8, 8))

    learn.export(RESULTS_DIR / "fastai_gas_sensor_model.pkl")
    return learn


def train_random_forest(df: pd.DataFrame) -> None:
    """Train and evaluate a Random Forest baseline model."""
    feature_columns = [column for column in df.columns if column.startswith("feature_")]

    X = df[feature_columns]
    y = df["target"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    accuracy_value = accuracy_score(y_valid, predictions)
    report = classification_report(y_valid, predictions)

    output = f"Random Forest Accuracy: {accuracy_value:.4f}\n\n{report}"
    print(output)

    with open(RESULTS_DIR / "random_forest_results.txt", "w", encoding="utf-8") as file:
        file.write(output)


def main() -> None:
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print(df.head())

    train_random_forest(df)
    train_fastai_model(df)


if __name__ == "__main__":
    main()
