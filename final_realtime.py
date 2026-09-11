import os
import time
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FATIGUE_FEATURE_FILE = (
    BASE_DIR / "imu_fatigue_features.csv"
)

MODEL_FILE = (
    BASE_DIR / "fatigue_model.pkl"
)

# ESP32 Wi-Fi Access Point
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

LIVE_MODE = True


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

# CURRENT HARDWARE
FATIGUE_SENSOR = "RH"
STABILITY_SENSOR = "RH"

# Future implementation
FUTURE_STABILITY_SENSORS = ["LF", "RF", "RL"]


# ============================================================
# SAMPLING & WINDOWING
# ============================================================

FS = 128
WINDOW_SECONDS = 5
WINDOW_SAMPLES = FS * WINDOW_SECONDS


# ============================================================
# LOAD FATIGUE DATA & MODEL
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING ML PIPELINE")
print("=" * 70)

if not os.path.exists(FATIGUE_FEATURE_FILE):
    raise FileNotFoundError(
        f"\nFatigue feature file not found:\n"
        f"{FATIGUE_FEATURE_FILE}"
    )

fatigue_df = pd.read_csv(
    FATIGUE_FEATURE_FILE
)

TARGET = "Class"

if TARGET not in fatigue_df.columns:
    raise ValueError(
        "Target column 'Class' not found."
    )


# ============================================================
# FATIGUE FEATURES
# ============================================================

REALTIME_FEATURES = [

    "acc_x_mean",
    "acc_x_std",
    "acc_x_min",
    "acc_x_max",
    "acc_x_range",
    "acc_x_rms",

    "acc_y_mean",
    "acc_y_std",
    "acc_y_min",
    "acc_y_max",
    "acc_y_range",
    "acc_y_rms",

    "acc_z_mean",
    "acc_z_std",
    "acc_z_min",
    "acc_z_max",
    "acc_z_range",
    "acc_z_rms",

    "ang_x_mean",
    "ang_x_std",
    "ang_x_min",
    "ang_x_max",
    "ang_x_range",
    "ang_x_rms",

    "ang_y_mean",
    "ang_y_std",
    "ang_y_min",
    "ang_y_max",
    "ang_y_range",
    "ang_y_rms",

    "ang_z_mean",
    "ang_z_std",
    "ang_z_min",
    "ang_z_max",
    "ang_z_range",
    "ang_z_rms",

    "acc_magnitude_mean",
    "acc_magnitude_std",
    "acc_magnitude_min",
    "acc_magnitude_max",
    "acc_magnitude_range",
    "acc_magnitude_rms",

    "ang_magnitude_mean",
    "ang_magnitude_std",
    "ang_magnitude_min",
    "ang_magnitude_max",
    "ang_magnitude_range",
    "ang_magnitude_rms",

    "duration_seconds",

    "fatigue_duration",
    "fatigue_acc_variability",
    "fatigue_acc_range",
    "fatigue_acc_rms",

    "fatigue_ang_variability",
    "fatigue_ang_range",
    "fatigue_ang_rms",
]


# Only use features actually present in training dataset
feature_columns = [
    f
    for f in REALTIME_FEATURES
    if f in fatigue_df.columns
]


print("\n" + "=" * 70)
print("REALTIME FATIGUE FEATURES")
print("=" * 70)

print(
    "\nNumber of features:",
    len(feature_columns)
)

for f in feature_columns:
    print(" -", f)


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train = (
    fatigue_df[feature_columns]
    .copy()
)

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.fillna(
    X_train.median()
)

y_train = fatigue_df[TARGET].copy()


print("\n" + "=" * 70)
print("FATIGUE MODEL")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Features:",
    X_train.shape[1]
)

print(
    "Classes:",
    sorted(y_train.unique())
)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

fatigue_model = None

if os.path.exists(MODEL_FILE):

    try:

        fatigue_model = joblib.load(
            MODEL_FILE
        )

        print(
            "\nExisting fatigue model loaded:"
        )

        print(
            MODEL_FILE
        )

    except Exception as e:

        print(
            "\nCould not load existing model."
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# TRAIN IF MODEL DOES NOT EXIST
# ============================================================

if fatigue_model is None:

    print(
        "\nTraining new Random Forest model..."
    )

    fatigue_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    fatigue_model.fit(
        X_train,
        y_train
    )

    try:

        os.makedirs(
            os.path.dirname(MODEL_FILE),
            exist_ok=True
        )

        joblib.dump(
            fatigue_model,
            MODEL_FILE
        )

        print(
            "\nNew model trained and saved:"
        )

        print(
            MODEL_FILE
        )

    except Exception as e:

        print(
            "\nCould not save model:",
            e
        )


print(
    "\nFatigue model ready."
)


# ============================================================
# REALTIME FEATURE EXTRACTION
# ============================================================

def extract_realtime_features(df):

    features = {}

    acc_x = (
        df["Accel X"]
        .to_numpy()
    )

    acc_y = (
        df["Accel Y"]
        .to_numpy()
    )

    acc_z = (
        df["Accel Z"]
        .to_numpy()
    )

    ang_x = (
        df["Gyro X"]
        .to_numpy()
    )

    ang_y = (
        df["Gyro Y"]
        .to_numpy()
    )

    ang_z = (
        df["Gyro Z"]
        .to_numpy()
    )


    # --------------------------------------------------------
    # FEATURE FUNCTION
    # --------------------------------------------------------

    def add_features(
        name,
        signal
    ):

        features[
            f"{name}_mean"
        ] = np.mean(signal)

        features[
            f"{name}_std"
        ] = np.std(signal)

        features[
            f"{name}_min"
        ] = np.min(signal)

        features[
            f"{name}_max"
        ] = np.max(signal)

        features[
            f"{name}_range"
        ] = np.ptp(signal)

        features[
            f"{name}_rms"
        ] = np.sqrt(
            np.mean(
                signal ** 2
            )
        )


    # --------------------------------------------------------
    # ACCELERATION
    # --------------------------------------------------------

    add_features(
        "acc_x",
        acc_x
    )

    add_features(
        "acc_y",
        acc_y
    )

    add_features(
        "acc_z",
        acc_z
    )


    # --------------------------------------------------------
    # ANGULAR VELOCITY
    # --------------------------------------------------------

    add_features(
        "ang_x",
        ang_x
    )

    add_features(
        "ang_y",
        ang_y
    )

    add_features(
        "ang_z",
        ang_z
    )


    # --------------------------------------------------------
    # MAGNITUDES
    # --------------------------------------------------------

    acc_magnitude = np.sqrt(
        acc_x ** 2 +
        acc_y ** 2 +
        acc_z ** 2
    )

    ang_magnitude = np.sqrt(
        ang_x ** 2 +
        ang_y ** 2 +
        ang_z ** 2
    )


    add_features(
        "acc_magnitude",
        acc_magnitude
    )

    add_features(
        "ang_magnitude",
        ang_magnitude
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    duration = (
        len(df) / FS
    )

    features[
        "duration_seconds"
    ] = duration

    features[
        "fatigue_duration"
    ] = duration


    # --------------------------------------------------------
    # FATIGUE FEATURES
    # --------------------------------------------------------

    features[
        "fatigue_acc_variability"
    ] = np.std(
        acc_magnitude
    )

    features[
        "fatigue_acc_range"
    ] = np.ptp(
        acc_magnitude
    )

    features[
        "fatigue_acc_rms"
    ] = np.sqrt(
        np.mean(
            acc_magnitude ** 2
        )
    )

    features[
        "fatigue_ang_variability"
    ] = np.std(
        ang_magnitude
    )

    features[
        "fatigue_ang_range"
    ] = np.ptp(
        ang_magnitude
    )

    features[
        "fatigue_ang_rms"
    ] = np.sqrt(
        np.mean(
            ang_magnitude ** 2
        )
    )


    return features


# ============================================================
# PREPARE FATIGUE INPUT
# ============================================================

def prepare_fatigue_input(
    realtime_features
):

    row = pd.DataFrame(
        [realtime_features]
    )

    row = row.reindex(
        columns=feature_columns
    )

    row = row.apply(
        pd.to_numeric,
        errors="coerce"
    )

    row = row.fillna(
        X_train.median()
    )

    return row


# ============================================================
# RH RULE-BASED STABILITY
# ============================================================

def calculate_rh_stability(df):

    """
    RH-only movement stability.

    Higher score = more stable.
    Lower score = more unstable.

    This is NOT physiological fatigue.
    It measures movement consistency
    from the RH IMU.
    """

    acc_x = (
        df["Accel X"]
        .to_numpy()
    )

    acc_y = (
        df["Accel Y"]
        .to_numpy()
    )

    acc_z = (
        df["Accel Z"]
        .to_numpy()
    )

    gyro_x = (
        df["Gyro X"]
        .to_numpy()
    )

    gyro_y = (
        df["Gyro Y"]
        .to_numpy()
    )

    gyro_z = (
        df["Gyro Z"]
        .to_numpy()
    )


    # --------------------------------------------------------
    # MAGNITUDES
    # --------------------------------------------------------

    acc_mag = np.sqrt(
        acc_x ** 2 +
        acc_y ** 2 +
        acc_z ** 2
    )

    gyro_mag = np.sqrt(
        gyro_x ** 2 +
        gyro_y ** 2 +
        gyro_z ** 2
    )


    # --------------------------------------------------------
    # VARIABILITY
    # --------------------------------------------------------

    acc_std = np.std(
        acc_mag
    )

    gyro_std = np.std(
        gyro_mag
    )


    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    acc_range = np.ptp(
        acc_mag
    )

    gyro_range = np.ptp(
        gyro_mag
    )


    # --------------------------------------------------------
    # JERK
    # --------------------------------------------------------

    if len(acc_mag) > 1:

        jerk = (
            np.diff(acc_mag)
            * FS
        )

        jerk_rms = np.sqrt(
            np.mean(
                jerk ** 2
            )
        )

    else:

        jerk_rms = 0


    # --------------------------------------------------------
    # NORMALIZED INSTABILITY COMPONENTS
    # --------------------------------------------------------

    acc_variability_score = (
        acc_std /
        (acc_std + 0.10)
    )

    acc_range_score = (
        acc_range /
        (acc_range + 0.50)
    )

    gyro_variability_score = (
        gyro_std /
        (gyro_std + 1.00)
    )

    gyro_range_score = (
        gyro_range /
        (gyro_range + 3.00)
    )

    jerk_score = (
        jerk_rms /
        (jerk_rms + 10.0)
    )


    # --------------------------------------------------------
    # WEIGHTED INSTABILITY
    # --------------------------------------------------------

    instability = (

        0.25 *
        acc_variability_score

        +

        0.15 *
        acc_range_score

        +

        0.25 *
        gyro_variability_score

        +

        0.15 *
        gyro_range_score

        +

        0.20 *
        jerk_score
    )


    instability = np.clip(
        instability,
        0,
        1
    )


    # --------------------------------------------------------
    # STABILITY SCORE
    # --------------------------------------------------------

    stability_score = (
        100 *
        (1 - instability)
    )

    stability_score = np.clip(
        stability_score,
        0,
        100
    )


    return stability_score


# ============================================================
# STABILITY LEVEL
# ============================================================

def get_stability_level(
    score
):

    if score >= 70:

        return "STABLE"

    elif score >= 45:

        return "MODERATE"

    else:

        return "UNSTABLE"


# ============================================================
# FATIGUE SCORE
# ============================================================

def calculate_fatigue_score(
    probability
):

    return np.clip(
        probability * 100,
        0,
        100
    )


# ============================================================
# FUSION
# ============================================================

def calculate_assistance(
    fatigue_probability,
    stability_score
):

    fatigue_score = (
        fatigue_probability * 100
    )

    instability_score = (
        100 - stability_score
    )


    # --------------------------------------------------------
    # FATIGUE + INSTABILITY FUSION
    # --------------------------------------------------------

    assistance_score = (

        0.60 *
        fatigue_score

        +

        0.40 *
        instability_score
    )


    assistance_score = np.clip(
        assistance_score,
        0,
        100
    )


    # --------------------------------------------------------
    # ASSIST-AS-NEEDED LEVELS
    # --------------------------------------------------------

    if assistance_score < 20:

        level = "OFF"

    elif assistance_score < 45:

        level = "LOW"

    elif assistance_score < 70:

        level = "MEDIUM"

    else:

        level = "HIGH"


    return (
        assistance_score,
        level
    )


# ============================================================
# CONTROLLER COMMAND
# ============================================================

def assistance_to_controller(
    level
):

    commands = {

        "OFF": 0.00,

        "LOW": 0.25,

        "MEDIUM": 0.50,

        "HIGH": 1.00
    }

    return commands[level]


# ============================================================
# WIFI DATA RECEIVING
# ============================================================

def get_rh_window(
    client_socket
):

    """
    ESP32 sends:

    RH,Timestamp,gx,gy,gz,ax,ay,az

    Example:

    RH,123456,0.12,-2.3,1.4,-0.06,0.59,0.79
    """

    buffer = []

    data_buffer = b""


    while len(buffer) < WINDOW_SAMPLES:

        try:

            data = client_socket.recv(
                4096
            )

            if not data:

                raise ConnectionError(
                    "ESP32 disconnected."
                )

            data_buffer += data


            while (
                b"\n" in data_buffer
                and
                len(buffer) < WINDOW_SAMPLES
            ):

                line, data_buffer = (
                    data_buffer.split(
                        b"\n",
                        1
                    )
                )

                line = (
                    line
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                    .strip()
                )


                if not line:

                    continue


                parts = line.split(",")


                # Expected:
                # RH + 7 numerical values

                if (
                    len(parts) != 8
                    or
                    parts[0] != "RH"
                ):

                    continue


                try:

                    values = list(
                        map(
                            float,
                            parts[1:]
                        )
                    )


                    row = {

                        "Time": values[0],

                        "Gyro X": values[1],
                        "Gyro Y": values[2],
                        "Gyro Z": values[3],

                        "Accel X": values[4],
                        "Accel Y": values[5],
                        "Accel Z": values[6]
                    }


                    buffer.append(
                        row
                    )


                except ValueError:

                    continue


        except socket.timeout:

            continue


    return pd.DataFrame(
        buffer
    )


# ============================================================
# SEND ASSISTANCE COMMAND TO ESP32
# ============================================================

def send_to_microcontroller(
    client_socket,
    assistance_score,
    assistance_level,
    controller_command
):

    message = (

        f"ASSIST,"
        f"{assistance_score:.2f},"
        f"{assistance_level},"
        f"{controller_command:.2f}\n"
    )


    client_socket.sendall(
        message.encode(
            "utf-8"
        )
    )


    print(
        "\n→ Sent to ESP32:",
        message.strip()
    )


# ============================================================
# PROCESS ONE WINDOW
# ============================================================

def process_window(
    rh_window,
    client_socket
):

    # ========================================================
    # FATIGUE ML
    # ========================================================

    realtime_features = (
        extract_realtime_features(
            rh_window
        )
    )


    X = (
        prepare_fatigue_input(
            realtime_features
        )
    )


    probabilities = (
        fatigue_model
        .predict_proba(X)[0]
    )


    classes = list(
        fatigue_model.classes_
    )


    # --------------------------------------------------------
    # CLASS 2 = HIGHER FATIGUE
    # --------------------------------------------------------

    if 2 in classes:

        fatigue_probability = (
            probabilities[
                classes.index(2)
            ]
        )

    else:

        fatigue_probability = (
            probabilities[-1]
        )


    fatigue_prediction = (
        fatigue_model
        .predict(X)[0]
    )


    fatigue_score = (
        calculate_fatigue_score(
            fatigue_probability
        )
    )


    # ========================================================
    # RH MOVEMENT STABILITY
    # ========================================================

    stability_score = (
        calculate_rh_stability(
            rh_window
        )
    )


    stability_level = (
        get_stability_level(
            stability_score
        )
    )


    # ========================================================
    # FUSION
    # ========================================================

    assistance_score, assistance_level = (
        calculate_assistance(
            fatigue_probability,
            stability_score
        )
    )


    controller_command = (
        assistance_to_controller(
            assistance_level
        )
    )


    # ========================================================
    # FULL OUTPUT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "REAL-TIME REHABILITATION STATUS"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # SENSOR CONFIGURATION
    # --------------------------------------------------------

    print(
        "\nSENSOR CONFIGURATION"
    )

    print(
        "  RH fatigue       : ACTIVE"
    )

    print(
        "  RH stability     : ACTIVE"
    )

    print(
        "  LF/RF/RL         : FUTURE IMPLEMENTATION"
    )


    print(
        "\nSAMPLING"
    )

    print(
        f"  Sampling rate    : {FS} Hz"
    )

    print(
        f"  Window           : {WINDOW_SECONDS} seconds"
    )

    print(
        f"  Samples          : {len(rh_window)}"
    )


    # --------------------------------------------------------
    # FATIGUE
    # --------------------------------------------------------

    print(
        "\nFATIGUE ML"
    )

    print(
        f"  Probability      : "
        f"{fatigue_probability * 100:.2f}%"
    )

    print(
        f"  Fatigue score    : "
        f"{fatigue_score:.2f}/100"
    )

    print(
        f"  Prediction       : "
        f"Class {fatigue_prediction}"
    )


    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    print(
        "\nRH MOVEMENT STABILITY"
    )

    print(
        f"  Stability score  : "
        f"{stability_score:.2f}/100"
    )

    print(
        f"  Stability level  : "
        f"{stability_level}"
    )


    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    print(
        "\nFUSION"
    )

    print(
        f"  Fatigue          : "
        f"{fatigue_score:.2f}/100"
    )

    print(
        f"  Stability        : "
        f"{stability_score:.2f}/100"
    )

    print(
        f"  Instability      : "
        f"{100 - stability_score:.2f}/100"
    )

    print(
        f"  Assistance       : "
        f"{assistance_score:.2f}/100"
    )

    print(
        f"  Assistance level : "
        f"{assistance_level}"
    )

    print(
        f"  Controller cmd   : "
        f"{controller_command:.2f}"
    )


    print("=" * 70)


    # ========================================================
    # SEND COMMAND
    # ========================================================

    send_to_microcontroller(
        client_socket,
        assistance_score,
        assistance_level,
        controller_command
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "REAL-TIME REHABILITATION ML PIPELINE"
    )

    print("=" * 70)


    print(
        "\nCommunication:"
    )

    print(
        "  ESP32 Wi-Fi Access Point"
    )

    print(
        f"  ESP32 IP        : {ESP32_IP}"
    )

    print(
        f"  TCP Port        : {ESP32_PORT}"
    )


    print(
        "\nCurrent sensor:"
    )

    print(
        "  RH = fatigue + movement stability"
    )


    print(
        "\nFuture sensors:"
    )

    print(
        "  LF + RF + RL = additional stability"
    )


    # ========================================================
    # LIVE MODE
    # ========================================================

    if LIVE_MODE:

        print("\n")

        print(
            f"Connecting to ESP32 at "
            f"{ESP32_IP}:{ESP32_PORT}..."
        )


        client_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )


        # Timeout allows the program to remain responsive
        client_socket.settimeout(2)


        try:

            client_socket.connect(
                (
                    ESP32_IP,
                    ESP32_PORT
                )
            )


            print(
                "\nConnected to ESP32 successfully."
            )

            print(
                "Receiving RH IMU data at 128 Hz."
            )

            print(
                "ML pipeline is now running."
            )


        except (
            socket.timeout,
            ConnectionRefusedError,
            OSError
        ) as e:

            print("\n")

            print("=" * 70)

            print(
                "ESP32 WIFI CONNECTION FAILED"
            )

            print("=" * 70)

            print(
                "\nCould not connect to:"
            )

            print(
                f"  {ESP32_IP}:{ESP32_PORT}"
            )

            print(
                "\nCheck that:"
            )

            print(
                "  1. ESP32 is powered."
            )

            print(
                "  2. Laptop is connected to ESP32_REHAB Wi-Fi."
            )

            print(
                "  3. ESP32 is running the Wi-Fi code."
            )

            print(
                "  4. IP address is 192.168.4.1."
            )

            print(
                "  5. TCP port is 5000."
            )

            print(
                "\nError:",
                e
            )

            client_socket.close()

            return


        try:

            while True:

                print(
                    f"\nWaiting for "
                    f"{WINDOW_SECONDS}-second RH window..."
                )


                rh_window = (
                    get_rh_window(
                        client_socket
                    )
                )


                if (
                    rh_window.empty
                    or
                    len(rh_window)
                    < WINDOW_SAMPLES
                ):

                    print(
                        "Incomplete RH window."
                    )

                    continue


                process_window(
                    rh_window,
                    client_socket
                )


        except KeyboardInterrupt:

            print(
                "\nStopping pipeline..."
            )


        except ConnectionError as e:

            print(
                "\nESP32 connection lost:",
                e
            )


        except OSError as e:

            print(
                "\nNetwork error:",
                e
            )


        finally:

            client_socket.close()

            print(
                "Wi-Fi connection closed."
            )


    # ========================================================
    # DEMO MODE
    # ========================================================

    else:

        print("\n")

        print("=" * 70)

        print(
            "DEMO MODE"
        )

        print("=" * 70)


        print(
            "\nGenerating simulated RH data."
        )

        print(
            "RH fatigue + RH stability + fusion"
        )

        print(
            "\nPress Ctrl+C to stop."
        )


        try:

            while True:

                # ------------------------------------------------
                # Generate demo data
                # ------------------------------------------------

                t = (
                    np.arange(
                        WINDOW_SAMPLES
                    ) / FS
                )


                fatigue_intensity = (
                    np.random.uniform(
                        0.2,
                        0.9
                    )
                )


                noise_level = (
                    np.random.uniform(
                        0.01,
                        0.15
                    )
                )


                acc_x = (
                    0.05 *
                    np.sin(
                        2 *
                        np.pi *
                        1.0 *
                        t
                    )
                )


                acc_y = (
                    0.05 *
                    np.sin(
                        2 *
                        np.pi *
                        1.2 *
                        t
                    )
                )


                acc_z = (
                    1.0 +
                    0.03 *
                    np.sin(
                        2 *
                        np.pi *
                        0.8 *
                        t
                    )
                )


                noise = (
                    np.random.normal(
                        0,
                        noise_level *
                        fatigue_intensity,
                        WINDOW_SAMPLES
                    )
                )


                acc_x += noise
                acc_y += noise
                acc_z += noise


                gyro_x = (
                    np.random.normal(
                        0,
                        0.5 *
                        fatigue_intensity,
                        WINDOW_SAMPLES
                    )
                )


                gyro_y = (
                    np.random.normal(
                        0,
                        0.5 *
                        fatigue_intensity,
                        WINDOW_SAMPLES
                    )
                )


                gyro_z = (
                    np.random.normal(
                        0,
                        0.5 *
                        fatigue_intensity,
                        WINDOW_SAMPLES
                    )
                )


                rh_window = pd.DataFrame({

                    "Time": t,

                    "Gyro X": gyro_x,
                    "Gyro Y": gyro_y,
                    "Gyro Z": gyro_z,

                    "Accel X": acc_x,
                    "Accel Y": acc_y,
                    "Accel Z": acc_z
                })


                # ------------------------------------------------
                # Process demo window
                # ------------------------------------------------

                realtime_features = (
                    extract_realtime_features(
                        rh_window
                    )
                )


                X = (
                    prepare_fatigue_input(
                        realtime_features
                    )
                )


                probabilities = (
                    fatigue_model
                    .predict_proba(X)[0]
                )


                classes = list(
                    fatigue_model.classes_
                )


                if 2 in classes:

                    fatigue_probability = (
                        probabilities[
                            classes.index(2)
                        ]
                    )

                else:

                    fatigue_probability = (
                        probabilities[-1]
                    )


                fatigue_prediction = (
                    fatigue_model
                    .predict(X)[0]
                )


                fatigue_score = (
                    calculate_fatigue_score(
                        fatigue_probability
                    )
                )


                stability_score = (
                    calculate_rh_stability(
                        rh_window
                    )
                )


                stability_level = (
                    get_stability_level(
                        stability_score
                    )
                )


                assistance_score, assistance_level = (
                    calculate_assistance(
                        fatigue_probability,
                        stability_score
                    )
                )


                controller_command = (
                    assistance_to_controller(
                        assistance_level
                    )
                )


                # ------------------------------------------------
                # Output
                # ------------------------------------------------

                print("\n")

                print("=" * 70)

                print(
                    "REAL-TIME REHABILITATION STATUS"
                )

                print("=" * 70)

                print(
                    "\nSENSOR CONFIGURATION"
                )

                print(
                    "  RH fatigue       : ACTIVE"
                )

                print(
                    "  RH stability     : ACTIVE"
                )

                print(
                    "  LF/RF/RL         : FUTURE IMPLEMENTATION"
                )

                print(
                    "\nSAMPLING"
                )

                print(
                    f"  Sampling rate    : {FS} Hz"
                )

                print(
                    f"  Window           : "
                    f"{WINDOW_SECONDS} seconds"
                )

                print(
                    f"  Samples          : "
                    f"{len(rh_window)}"
                )

                print(
                    "\nFATIGUE ML"
                )

                print(
                    f"  Probability      : "
                    f"{fatigue_probability * 100:.2f}%"
                )

                print(
                    f"  Fatigue score    : "
                    f"{fatigue_score:.2f}/100"
                )

                print(
                    f"  Prediction       : "
                    f"Class {fatigue_prediction}"
                )

                print(
                    "\nRH MOVEMENT STABILITY"
                )

                print(
                    f"  Stability score  : "
                    f"{stability_score:.2f}/100"
                )

                print(
                    f"  Stability level  : "
                    f"{stability_level}"
                )

                print(
                    "\nFUSION"
                )

                print(
                    f"  Fatigue          : "
                    f"{fatigue_score:.2f}/100"
                )

                print(
                    f"  Stability        : "
                    f"{stability_score:.2f}/100"
                )

                print(
                    f"  Instability      : "
                    f"{100 - stability_score:.2f}/100"
                )

                print(
                    f"  Assistance       : "
                    f"{assistance_score:.2f}/100"
                )

                print(
                    f"  Assistance level : "
                    f"{assistance_level}"
                )

                print(
                    f"  Controller cmd   : "
                    f"{controller_command:.2f}"
                )

                print(
                    "=" * 70
                )


                time.sleep(1)


        except KeyboardInterrupt:

            print(
                "\nDemo stopped."
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()