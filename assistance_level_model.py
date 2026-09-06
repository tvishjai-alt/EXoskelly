
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# IMU ASSISTANCE LEVEL MODEL
# Movement classification + movement degradation
# ============================================================

print("=" * 70)
print("IMU ASSISTANCE LEVEL MODEL")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_DIR = Path(r"F:\SIH\FINAL_PIPELINE\data")
FATIGUE_DIR = DATA_DIR / "fatigue_analysis"

FATIGUE_FILE = FATIGUE_DIR / "imu_fatigue_clean_predictions.csv"
MOVEMENT_FILE = DATA_DIR / "imu_features.csv"

OUTPUT_FILE = FATIGUE_DIR / "imu_assistance_levels.csv"

# ------------------------------------------------------------
# LOAD FATIGUE PREDICTIONS
# ------------------------------------------------------------

print("\nLoading fatigue predictions:")
print(FATIGUE_FILE)

fatigue_df = pd.read_csv(FATIGUE_FILE)

print(f"Rows: {len(fatigue_df)}")
print(f"Columns: {len(fatigue_df.columns)}")

print("\nColumns:")
print(fatigue_df.columns.tolist())

# ------------------------------------------------------------
# IDENTIFY FATIGUE PREDICTION COLUMN
# ------------------------------------------------------------

possible_fatigue_columns = [
    "Predicted_Fatigue_Level",
    "Predicted_Fatigue",
    "Fatigue_Level",
    "Predicted_Class",
    "Prediction"
]

fatigue_column = None

for col in possible_fatigue_columns:
    if col in fatigue_df.columns:
        fatigue_column = col
        break

if fatigue_column is None:
    raise ValueError(
        "Could not identify fatigue prediction column.\n"
        "Available columns:\n"
        + "\n".join(fatigue_df.columns)
    )

print(f"\nUsing fatigue prediction column: {fatigue_column}")

# ------------------------------------------------------------
# MAP FATIGUE LEVEL
# ------------------------------------------------------------

def map_fatigue_level(value):

    if isinstance(value, str):

        value_lower = value.lower()

        if "low" in value_lower:
            return 0

        if "medium" in value_lower:
            return 1

        if "high" in value_lower:
            return 2

    try:
        value = int(value)

        if value in [0, 1, 2]:
            return value

    except:
        pass

    return np.nan


fatigue_df["Fatigue_Level"] = fatigue_df[fatigue_column].apply(
    map_fatigue_level
)

if fatigue_df["Fatigue_Level"].isna().any():

    print("\nWARNING: Some fatigue levels could not be interpreted.")

    print(
        fatigue_df[
            fatigue_df["Fatigue_Level"].isna()
        ][fatigue_column].value_counts()
    )

    fatigue_df = fatigue_df.dropna(
        subset=["Fatigue_Level"]
    )

fatigue_df["Fatigue_Level"] = (
    fatigue_df["Fatigue_Level"]
    .astype(int)
)

# ------------------------------------------------------------
# ASSISTANCE LOGIC
# ------------------------------------------------------------

def determine_assistance(fatigue_level):

    if fatigue_level == 0:
        return "LOW"

    elif fatigue_level == 1:
        return "MEDIUM"

    elif fatigue_level == 2:
        return "HIGH"

    return "LOW"


fatigue_df["Assistance_Level"] = fatigue_df[
    "Fatigue_Level"
].apply(determine_assistance)

# ------------------------------------------------------------
# ASSISTANCE NUMERIC VALUE
# ------------------------------------------------------------

assistance_numeric = {
    "OFF": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}

fatigue_df["Assistance_Value"] = fatigue_df[
    "Assistance_Level"
].map(assistance_numeric)

# ------------------------------------------------------------
# ADD MOVEMENT INFORMATION IF AVAILABLE
# ------------------------------------------------------------

if "Class" in fatigue_df.columns:

    def movement_name(value):

        try:
            value = int(value)

            if value == 1:
                return "Flexion"

            elif value == 2:
                return "Extension"

        except:
            pass

        return "Unknown"

    fatigue_df["Movement_Type"] = fatigue_df[
        "Class"
    ].apply(movement_name)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ASSISTANCE LEVEL DISTRIBUTION")
print("=" * 70)

print(
    fatigue_df["Assistance_Level"]
    .value_counts()
    .sort_index()
)

print("\nPercentage distribution:")

percentage = (
    fatigue_df["Assistance_Level"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for level, value in percentage.items():

    print(
        f"{level:8s}: {value:.2f}%"
    )

# ------------------------------------------------------------
# FATIGUE × ASSISTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FATIGUE LEVEL → ASSISTANCE LEVEL")
print("=" * 70)

print(
    pd.crosstab(
        fatigue_df["Fatigue_Level"],
        fatigue_df["Assistance_Level"]
    )
)

# ------------------------------------------------------------
# MOVEMENT × ASSISTANCE
# ------------------------------------------------------------

if "Movement_Type" in fatigue_df.columns:

    print("\n" + "=" * 70)
    print("MOVEMENT TYPE × ASSISTANCE")
    print("=" * 70)

    print(
        pd.crosstab(
            fatigue_df["Movement_Type"],
            fatigue_df["Assistance_Level"]
        )
    )

# ------------------------------------------------------------
# SUBJECT-WISE ASSISTANCE
# ------------------------------------------------------------

if "Subject" in fatigue_df.columns:

    print("\n" + "=" * 70)
    print("SUBJECT-WISE ASSISTANCE")
    print("=" * 70)

    subject_assistance = pd.crosstab(
        fatigue_df["Subject"],
        fatigue_df["Assistance_Level"],
        normalize="index"
    ) * 100

    print(
        subject_assistance.round(2)
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

fatigue_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("ASSISTANCE MODEL COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nAssistance mapping:")
print("Fatigue Level 0 → LOW")
print("Fatigue Level 1 → MEDIUM")
print("Fatigue Level 2 → HIGH")

print("\nIMPORTANT:")
print(
    "This is an initial assistance policy based on "
    "movement-degradation level."
)

print(
    "It is NOT a direct measurement of physiological fatigue "
    "and should be validated before real-world actuator control."
)
