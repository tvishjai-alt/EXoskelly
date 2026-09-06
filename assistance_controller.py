
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ASSISTANCE DECISION LAYER
# Converts fatigue/degradation predictions into:
# OFF -> LOW -> MEDIUM -> HIGH
#
# Includes:
#   - Temporal smoothing
#   - Hysteresis
#   - Safety limits
#   - Assistance transition tracking
#
# IMPORTANT:
# This does NOT directly control a motor.
# ============================================================

print("=" * 70)
print("IMU ASSISTANCE DECISION LAYER")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_DIR = Path(r"F:\SIH\FINAL_PIPELINE\data")
FATIGUE_DIR = DATA_DIR / "fatigue_analysis"

INPUT_FILE = FATIGUE_DIR / "imu_assistance_levels.csv"
OUTPUT_FILE = FATIGUE_DIR / "imu_assistance_controller_output.csv"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# SORT CHRONOLOGICALLY
# ------------------------------------------------------------

df = df.sort_values(
    ["Subject", "Run", "Segment"]
).reset_index(drop=True)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Number of recent predictions used for smoothing
SMOOTHING_WINDOW = 3

# Number of consecutive predictions required
# before changing assistance level
HYSTERESIS_COUNT = 2

# Assistance levels
OFF = 0
LOW = 1
MEDIUM = 2
HIGH = 3

LEVEL_NAMES = {
    0: "OFF",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH"
}

# ------------------------------------------------------------
# INITIAL FATIGUE → ASSISTANCE MAPPING
# ------------------------------------------------------------

def fatigue_to_assistance(fatigue_level):

    if fatigue_level == 0:
        return LOW

    elif fatigue_level == 1:
        return MEDIUM

    elif fatigue_level == 2:
        return HIGH

    return LOW


df["Raw_Assistance_Value"] = df[
    "Predicted_Fatigue_Level"
].apply(fatigue_to_assistance)

df["Raw_Assistance_Level"] = df[
    "Raw_Assistance_Value"
].map(LEVEL_NAMES)

# ------------------------------------------------------------
# TEMPORAL SMOOTHING
# ------------------------------------------------------------

print("\nApplying temporal smoothing...")

df["Smoothed_Fatigue_Level"] = (
    df.groupby(["Subject", "Run"])["Predicted_Fatigue_Level"]
    .transform(
        lambda x: x.rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1
        ).median()
    )
    .round()
    .astype(int)
)

df["Smoothed_Assistance_Value"] = df[
    "Smoothed_Fatigue_Level"
].apply(fatigue_to_assistance)

df["Smoothed_Assistance_Level"] = df[
    "Smoothed_Assistance_Value"
].map(LEVEL_NAMES)

# ------------------------------------------------------------
# HYSTERESIS
# ------------------------------------------------------------

print("Applying hysteresis...")

final_levels = []

current_level = LOW
candidate_level = LOW
candidate_count = 0

previous_subject = None
previous_run = None

for _, row in df.iterrows():

    subject = row["Subject"]
    run = row["Run"]

    requested_level = int(
        row["Smoothed_Assistance_Value"]
    )

    # Reset controller at beginning of every run
    if (
        subject != previous_subject
        or run != previous_run
    ):

        current_level = LOW
        candidate_level = LOW
        candidate_count = 0

    # --------------------------------------------------------
    # Same as current level
    # --------------------------------------------------------

    if requested_level == current_level:

        candidate_level = current_level
        candidate_count = 0

    # --------------------------------------------------------
    # Different requested level
    # --------------------------------------------------------

    else:

        if requested_level == candidate_level:

            candidate_count += 1

        else:

            candidate_level = requested_level
            candidate_count = 1

        # Change level only after consecutive confirmation
        if candidate_count >= HYSTERESIS_COUNT:

            current_level = candidate_level
            candidate_count = 0

    final_levels.append(current_level)

    previous_subject = subject
    previous_run = run

df["Final_Assistance_Value"] = final_levels

df["Final_Assistance_Level"] = df[
    "Final_Assistance_Value"
].map(LEVEL_NAMES)

# ------------------------------------------------------------
# SAFETY LIMIT
# ------------------------------------------------------------

# Current research prototype allows HIGH as maximum.
# This does NOT represent a clinically validated torque limit.

MAX_ASSISTANCE = HIGH

df["Final_Assistance_Value"] = np.minimum(
    df["Final_Assistance_Value"],
    MAX_ASSISTANCE
)

df["Final_Assistance_Level"] = df[
    "Final_Assistance_Value"
].map(LEVEL_NAMES)

# ------------------------------------------------------------
# DETECT TRANSITIONS
# ------------------------------------------------------------

df["Previous_Assistance_Value"] = (
    df.groupby(["Subject", "Run"])
    ["Final_Assistance_Value"]
    .shift(1)
)

df["Assistance_Transition"] = (
    df["Final_Assistance_Value"]
    != df["Previous_Assistance_Value"]
)

# First segment of every run is not considered a transition
df.loc[
    df["Previous_Assistance_Value"].isna(),
    "Assistance_Transition"
] = False

# ------------------------------------------------------------
# TRANSITION TYPE
# ------------------------------------------------------------

def transition_type(row):

    if not row["Assistance_Transition"]:
        return "NONE"

    previous = int(
        row["Previous_Assistance_Value"]
    )

    current = int(
        row["Final_Assistance_Value"]
    )

    if current > previous:
        return "INCREASE"

    elif current < previous:
        return "DECREASE"

    return "NONE"


df["Transition_Type"] = df.apply(
    transition_type,
    axis=1
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ASSISTANCE DISTRIBUTION")
print("=" * 70)

distribution = (
    df["Final_Assistance_Level"]
    .value_counts()
    .reindex(
        ["OFF", "LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )
)

print(distribution)

print("\nPercentage:")

percentage = (
    distribution / len(df) * 100
)

for level in [
    "OFF",
    "LOW",
    "MEDIUM",
    "HIGH"
]:

    print(
        f"{level:8s}: "
        f"{percentage[level]:.2f}%"
    )

# ------------------------------------------------------------
# RAW VS FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RAW VS FINAL ASSISTANCE")
print("=" * 70)

comparison = pd.crosstab(
    df["Raw_Assistance_Level"],
    df["Final_Assistance_Level"]
)

print(comparison)

# ------------------------------------------------------------
# TRANSITIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ASSISTANCE TRANSITIONS")
print("=" * 70)

transition_counts = (
    df["Transition_Type"]
    .value_counts()
)

print(transition_counts)

print(
    f"\nTotal transitions: "
    f"{df['Assistance_Transition'].sum()}"
)

# ------------------------------------------------------------
# SUBJECT-WISE RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUBJECT-WISE FINAL ASSISTANCE")
print("=" * 70)

subject_table = pd.crosstab(
    df["Subject"],
    df["Final_Assistance_Level"],
    normalize="index"
) * 100

subject_table = subject_table.reindex(
    columns=["OFF", "LOW", "MEDIUM", "HIGH"],
    fill_value=0
)

print(
    subject_table.round(2)
)

# ------------------------------------------------------------
# MOVEMENT-WISE RESULTS
# ------------------------------------------------------------

if "Movement_Type" in df.columns:

    print("\n" + "=" * 70)
    print("MOVEMENT × FINAL ASSISTANCE")
    print("=" * 70)

    movement_table = pd.crosstab(
        df["Movement_Type"],
        df["Final_Assistance_Level"],
        normalize="index"
    ) * 100

    movement_table = movement_table.reindex(
        columns=["OFF", "LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )

    print(
        movement_table.round(2)
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("ASSISTANCE DECISION LAYER COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nControl logic:")
print("Fatigue prediction")
print("        ↓")
print("Temporal smoothing")
print("        ↓")
print("Hysteresis")
print("        ↓")
print("OFF / LOW / MEDIUM / HIGH")
print("        ↓")
print("Future PID controller")
print("        ↓")
print("Future motor command")

print("\nIMPORTANT:")
print(
    "This output is a research-level assistance policy."
)

print(
    "It must not be connected directly to a physical actuator "
    "without independent safety limits, mechanical safeguards, "
    "and clinical validation."
)

