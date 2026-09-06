import numpy as np
import pandas as pd
import os

# ============================================================
# PATH
# ============================================================

BASE = r"F:\SIH\DATASETS"

# CHANGE THIS if your actual folder is different
OUTPUT = r"F:\SIH\TESTING_PIPELINE\joint_stability_features.csv"

# ============================================================
# CHANNEL NAMES
# ============================================================

# Your files currently have 6 channels.
# We don't yet know the exact 6-channel configuration,
# so we'll treat them generically.

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal):

    features = {}

    # signal shape = (880, 6)

    for ch in range(signal.shape[1]):

        x = signal[:, ch]

        # Basic statistics
        features[f"ch{ch}_mean"] = np.mean(x)
        features[f"ch{ch}_std"] = np.std(x)
        features[f"ch{ch}_rms"] = np.sqrt(np.mean(x ** 2))
        features[f"ch{ch}_max"] = np.max(x)
        features[f"ch{ch}_min"] = np.min(x)
        features[f"ch{ch}_range"] = np.ptp(x)

        # Median absolute deviation
        features[f"ch{ch}_mad"] = np.median(
            np.abs(x - np.median(x))
        )

        # Jerk / rate of change
        diff = np.diff(x)

        features[f"ch{ch}_jerk_rms"] = np.sqrt(
            np.mean(diff ** 2)
        )

        # Movement energy
        features[f"ch{ch}_energy"] = np.mean(x ** 2)

    # --------------------------------------------------------
    # Overall signal magnitude
    # --------------------------------------------------------

    magnitude = np.sqrt(
        np.sum(signal ** 2, axis=1)
    )

    features["magnitude_mean"] = np.mean(magnitude)
    features["magnitude_std"] = np.std(magnitude)
    features["magnitude_rms"] = np.sqrt(
        np.mean(magnitude ** 2)
    )
    features["magnitude_range"] = np.ptp(magnitude)

    # --------------------------------------------------------
    # Smoothness
    # --------------------------------------------------------

    second_diff = np.diff(signal, n=2, axis=0)

    features["smoothness"] = np.mean(
        second_diff ** 2
    )

    return features


# ============================================================
# PROCESS ALL MOVEMENTS
# ============================================================

all_features = []

for movement in range(16):

    movement_id = f"{movement:03d}"

    for sensor in [1, 2]:

        filename = f"{movement_id}_{sensor}.npy"

        filepath = os.path.join(
            BASE,
            filename
        )

        if not os.path.exists(filepath):

            print("MISSING:", filename)
            continue

        try:

            data = np.load(filepath)

            print(
                f"{filename} → {data.shape}"
            )

            # ------------------------------------------------
            # Each row = one movement sample
            # ------------------------------------------------

            for sample_idx in range(data.shape[0]):

                signal = data[sample_idx]

                features = extract_features(
                    signal
                )

                features["movement"] = movement
                features["sensor"] = sensor
                features["sample"] = sample_idx

                all_features.append(features)

        except Exception as e:

            print(
                f"ERROR: {filename}: {e}"
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    all_features
)

print("\n" + "=" * 60)
print("JOINT STABILITY FEATURE DATASET")
print("=" * 60)

print(
    "\nShape:",
    features_df.shape
)

print(
    "\nMovements:",
    features_df["movement"].nunique()
)

print(
    "\nSamples per movement:"
)

print(
    features_df["movement"].value_counts().sort_index()
)


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT,
    index=False
)

print(
    "\nSaved to:",
    OUTPUT
)