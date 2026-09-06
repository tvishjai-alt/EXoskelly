import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = r"F:\SIH\FINAL_PIPELINE\data\imu_fatigue_features.csv"

OUTPUT_DIR = r"F:\SIH\FINAL_PIPELINE\data\fatigue_analysis"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "imu_fatigue_clean_model_results.csv"
)

IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "imu_fatigue_clean_feature_importance.csv"
)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "imu_fatigue_clean_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CLEAN IMU FATIGUE / MOVEMENT-DEGRADATION MODEL")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ============================================================
# REMOVE BASELINE
# ============================================================

df = df[df["Class"] != 0].copy()

print("\nRows after removing baseline:")
print(len(df))


# ============================================================
# CREATE FATIGUE LEVELS
# ============================================================

print("\nCreating fatigue/degradation levels...")

# Use the movement degradation score only to define the target.
# IMPORTANT:
# movement_degradation_score itself will NOT be used as an ML feature.

df["Fatigue_Level"] = pd.qcut(
    df["movement_degradation_score"],
    q=3,
    labels=[0, 1, 2]
)

df["Fatigue_Level"] = df["Fatigue_Level"].astype(int)

print("\nFatigue-level distribution:")
print(df["Fatigue_Level"].value_counts().sort_index())


# ============================================================
# DEFINE FEATURES
# ============================================================

# Metadata / target / progression-derived variables that should
# NOT be given to the model.

EXCLUDED_COLUMNS = [
    "Subject",
    "Run",
    "Segment",
    "Class",
    "Movement",
    "SMR",
    "Repetition",

    # Target
    "movement_degradation_score",
    "Fatigue_Level",

    # Explicit progression variables
    "Run_Progress",
    "Run_Segment_Index",
    "Run_Total_Segments",

    # Start/end sample information
    "Start_Sample",
    "End_Sample",
    "Num_Samples",

    # Features that explicitly encode progression/change
    "fatigue_duration_change",
    "fatigue_acc_variability_change",
    "fatigue_acc_range_change",
    "fatigue_ang_variability_change",
    "fatigue_ang_range_change",
]


feature_columns = [
    col for col in df.columns
    if col not in EXCLUDED_COLUMNS
]

print("\nNumber of ML features:")
print(len(feature_columns))

print("\nFeatures:")
for feature in feature_columns:
    print(f"  - {feature}")


# ============================================================
# DATA
# ============================================================

X = df[feature_columns].copy()
y = df["Fatigue_Level"].copy()

subjects = df["Subject"].values


# Make sure all features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using column medians
X = X.fillna(X.median())


# ============================================================
# SUBJECT-WISE LEAVE-ONE-SUBJECT-OUT VALIDATION
# ============================================================

unique_subjects = sorted(df["Subject"].unique())

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

all_true = []
all_pred = []

fold_results = []


for fold_number, test_subject in enumerate(
    unique_subjects,
    start=1
):

    train_mask = subjects != test_subject
    test_mask = subjects == test_subject

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print(
        f"\nFold {fold_number}/{len(unique_subjects)} "
        f"| Test subject: {test_subject}"
    )

    print(
        "Training subjects:",
        [
            s for s in unique_subjects
            if s != test_subject
        ]
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Fold accuracy: {accuracy:.4f}"
    )

    fold_results.append({
        "Test_Subject": test_subject,
        "Accuracy": accuracy
    })

    all_true.extend(y_test)
    all_pred.extend(predictions)


# ============================================================
# OVERALL RESULTS
# ============================================================

all_true = np.array(all_true)
all_pred = np.array(all_pred)

fold_results_df = pd.DataFrame(
    fold_results
)

mean_accuracy = fold_results_df["Accuracy"].mean()
std_accuracy = fold_results_df["Accuracy"].std()

overall_accuracy = accuracy_score(
    all_true,
    all_pred
)

precision = precision_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "-" * 70)
print("OVERALL RESULTS")
print("-" * 70)

print(
    f"Mean fold accuracy : {mean_accuracy:.4f}"
)

print(
    f"Std fold accuracy  : {std_accuracy:.4f}"
)

print(
    f"Overall accuracy   : {overall_accuracy:.4f}"
)

print(
    f"Precision          : {precision:.4f}"
)

print(
    f"Recall             : {recall:.4f}"
)

print(
    f"F1-score           : {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred
)

print("\nConfusion Matrix")
print("(Rows = actual, Columns = predicted)")
print()
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")

print(
    classification_report(
        all_true,
        all_pred,
        target_names=[
            "Low",
            "Medium",
            "High"
        ],
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Train final model on all subjects ONLY for feature importance.
# This model is not used to calculate validation accuracy.

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_model.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print("\nTop 20 features:")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = df[
    [
        "Subject",
        "Run",
        "Segment",
        "Class",
        "Fatigue_Level"
    ]
].copy()

prediction_df["Predicted_Fatigue_Level"] = all_pred

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results_summary = pd.DataFrame({
    "Metric": [
        "Mean Fold Accuracy",
        "Std Fold Accuracy",
        "Overall Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        mean_accuracy,
        std_accuracy,
        overall_accuracy,
        precision,
        recall,
        f1
    ]
})

results_summary.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CLEAN IMU FATIGUE MODEL COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULTS_FILE)

print(IMPORTANCE_FILE)

print(PREDICTIONS_FILE)

print("\nIMPORTANT:")
print(
    "The model predicts movement-degradation levels "
    "as a fatigue proxy."
)

print(
    "It does NOT directly measure physiological fatigue."
)

print(
    "\nProgression and target-derived features were excluded "
    "from the ML input."
)