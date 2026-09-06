import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# ============================================================
# PATHS
# ============================================================

INPUT = r"F:\SIH\TESTING_PIPELINE\joint_stability_features.csv"

OUTPUT = r"F:\SIH\TESTING_PIPELINE\joint_stability_scores.csv"


# ============================================================
# LOAD FEATURES
# ============================================================

df = pd.read_csv(INPUT)

print("Input dataset:", df.shape)


# ============================================================
# FEATURES RELATED TO MOVEMENT INSTABILITY
# ============================================================

# Higher values generally indicate more variability,
# abrupt movement, or poorer smoothness.

instability_features = [
    c for c in df.columns
    if (
        "_std" in c
        or "_range" in c
        or "_jerk_rms" in c
        or c == "magnitude_std"
        or c == "magnitude_range"
        or c == "smoothness"
    )
]

print("\nInstability features:")
print(len(instability_features))

print(instability_features)


# ============================================================
# HANDLE MISSING / INFINITE VALUES
# ============================================================

X = df[instability_features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# ROBUST SCALING
# ============================================================

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# CALCULATE INSTABILITY INDEX
# ============================================================

# Mean normalized instability across features

instability_index = np.mean(
    X_scaled,
    axis=1
)


# ============================================================
# NORMALIZE INSTABILITY TO 0–100
# ============================================================

# Percentile-based normalization prevents extreme
# outliers from completely dominating the score.

low = np.percentile(
    instability_index,
    5
)

high = np.percentile(
    instability_index,
    95
)

instability_clipped = np.clip(
    instability_index,
    low,
    high
)

instability_0_100 = (
    (instability_clipped - low)
    /
    (high - low)
) * 100


# ============================================================
# STABILITY SCORE
# ============================================================

# High instability = low stability

stability_score = (
    100 - instability_0_100
)

stability_score = np.clip(
    stability_score,
    0,
    100
)


# ============================================================
# ADD SCORES
# ============================================================

df["instability_index"] = instability_index

df["stability_score"] = stability_score


# ============================================================
# STABILITY CATEGORY
# ============================================================

def classify_stability(score):

    if score >= 75:
        return "Stable"

    elif score >= 50:
        return "Moderate"

    elif score >= 25:
        return "Unstable"

    else:
        return "Highly Unstable"


df["stability_category"] = (
    df["stability_score"]
    .apply(classify_stability)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("JOINT STABILITY RESULTS")
print("=" * 60)

print(
    "\nStability score statistics:"
)

print(
    df["stability_score"].describe()
)

print(
    "\nStability categories:"
)

print(
    df["stability_category"].value_counts()
)


# ============================================================
# MOVEMENT-WISE RESULTS
# ============================================================

print(
    "\nAverage stability by movement:"
)

movement_scores = (
    df.groupby("movement")["stability_score"]
    .agg(["mean", "std", "count"])
    .round(2)
)

print(
    movement_scores
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print(
    "\nSaved to:"
)

print(OUTPUT)