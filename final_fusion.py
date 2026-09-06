import numpy as np


# ============================================================
# FINAL REHABILITATION FUSION MODULE
#
# Fatigue + Joint Stability
#          ↓
# Combined Rehabilitation State
#          ↓
# Assistance Level
#          ↓
# High-Level Controller
#          ↓
# PID Controller
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

FATIGUE_WEIGHT = 0.40
INSTABILITY_WEIGHT = 0.60


# ============================================================
# 1. FATIGUE LEVEL
# ============================================================

def classify_fatigue(fatigue_probability):

    """
    fatigue_probability:
        0.0 → no fatigue
        1.0 → high fatigue

    Returns:
        LOW / MEDIUM / HIGH
    """

    if fatigue_probability < 0.33:
        return "LOW"

    elif fatigue_probability < 0.66:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# 2. STABILITY LEVEL
# ============================================================

def classify_stability(stability_score):

    """
    stability_score:
        100 → highly stable
        0   → highly unstable
    """

    if stability_score >= 75:
        return "STABLE"

    elif stability_score >= 50:
        return "MODERATE"

    elif stability_score >= 25:
        return "UNSTABLE"

    else:
        return "HIGHLY_UNSTABLE"


# ============================================================
# 3. FUSION
# ============================================================

def calculate_assistance(
    fatigue_probability,
    stability_score
):

    """
    Combines fatigue and joint stability.

    Parameters
    ----------
    fatigue_probability : float
        0 → no fatigue
        1 → high fatigue

    stability_score : float
        0 → unstable
        100 → stable


    Returns
    -------
    dictionary containing:

        fatigue_level
        stability_level
        combined_rehab_state
        assistance_score
        assistance_level
    """


    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not 0 <= fatigue_probability <= 1:

        raise ValueError(
            "fatigue_probability must be between 0 and 1"
        )


    if not 0 <= stability_score <= 100:

        raise ValueError(
            "stability_score must be between 0 and 100"
        )


    # --------------------------------------------------------
    # FATIGUE
    # --------------------------------------------------------

    fatigue_score = fatigue_probability * 100

    fatigue_level = classify_fatigue(
        fatigue_probability
    )


    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    instability_score = 100 - stability_score

    stability_level = classify_stability(
        stability_score
    )


    # --------------------------------------------------------
    # FUSION
    #
    # Higher fatigue       → more assistance
    # Higher instability   → more assistance
    #
    # --------------------------------------------------------

    assistance_score = (

        FATIGUE_WEIGHT * fatigue_score

        +

        INSTABILITY_WEIGHT * instability_score

    )


    assistance_score = float(
        np.clip(
            assistance_score,
            0,
            100
        )
    )


    # --------------------------------------------------------
    # ASSISTANCE LEVEL
    # --------------------------------------------------------

    if assistance_score < 25:

        assistance_level = "OFF"

    elif assistance_score < 50:

        assistance_level = "LOW"

    elif assistance_score < 75:

        assistance_level = "MEDIUM"

    else:

        assistance_level = "HIGH"


    # --------------------------------------------------------
    # COMBINED REHABILITATION STATE
    # --------------------------------------------------------

    combined_rehab_state = (

        f"Fatigue={fatigue_level}, "

        f"Stability={stability_level}"

    )


    return {

        "fatigue_probability":
            fatigue_probability,

        "fatigue_score":
            fatigue_score,

        "fatigue_level":
            fatigue_level,

        "stability_score":
            stability_score,

        "instability_score":
            instability_score,

        "stability_level":
            stability_level,

        "assistance_score":
            assistance_score,

        "assistance_level":
            assistance_level,

        "combined_rehab_state":
            combined_rehab_state

    }


# ============================================================
# 4. HIGH-LEVEL CONTROLLER INTERFACE
# ============================================================

def high_level_controller(
    fatigue_probability,
    stability_score
):

    """
    Interface between ML/fusion system
    and the high-level rehabilitation controller.
    """

    result = calculate_assistance(
        fatigue_probability,
        stability_score
    )


    # --------------------------------------------------------
    # Controller command
    # --------------------------------------------------------

    assistance_level = result[
        "assistance_level"
    ]

    assistance_score = result[
        "assistance_score"
    ]


    # --------------------------------------------------------
    # Convert assistance level into normalized
    # controller command
    #
    # OFF    = 0.00
    # LOW    = 0.25
    # MEDIUM = 0.50
    # HIGH   = 1.00
    # --------------------------------------------------------

    assistance_command = {

        "OFF": 0.00,

        "LOW": 0.25,

        "MEDIUM": 0.50,

        "HIGH": 1.00

    }[assistance_level]


    result[
        "assistance_command"
    ] = assistance_command


    return result


# ============================================================
# 5. TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "FINAL FATIGUE + JOINT STABILITY FUSION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # EXAMPLE INPUTS
    #
    # Replace these with actual model outputs later.
    # --------------------------------------------------------

    fatigue_probability = 0.70

    stability_score = 40.0


    result = high_level_controller(

        fatigue_probability,

        stability_score

    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()

    print(
        f"Fatigue probability : "
        f"{result['fatigue_probability']:.3f}"
    )

    print(
        f"Fatigue score       : "
        f"{result['fatigue_score']:.2f}/100"
    )

    print(
        f"Fatigue level       : "
        f"{result['fatigue_level']}"
    )

    print()

    print(
        f"Stability score     : "
        f"{result['stability_score']:.2f}/100"
    )

    print(
        f"Stability level     : "
        f"{result['stability_level']}"
    )

    print()

    print(
        f"Combined state      : "
        f"{result['combined_rehab_state']}"
    )

    print()

    print(
        f"Assistance score    : "
        f"{result['assistance_score']:.2f}/100"
    )

    print(
        f"Assistance level    : "
        f"{result['assistance_level']}"
    )

    print(
        f"Controller command  : "
        f"{result['assistance_command']:.2f}"
    )

    print()

    print("=" * 70)

    print("FUSION COMPLETE")

    print("=" * 70)