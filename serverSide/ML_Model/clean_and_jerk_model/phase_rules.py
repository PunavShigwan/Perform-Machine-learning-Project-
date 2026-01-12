"""
Rule-based judging for Clean & Jerk
Phase-1: Deterministic rules
"""

class CleanJerkRules:

    @staticmethod
    def check_elbow_lock(elbow_angle):
        return elbow_angle >= 170

    @staticmethod
    def check_knee_extension(knee_angle):
        return knee_angle >= 160

    @staticmethod
    def check_stability(stability_time):
        return stability_time >= 1.0  # seconds

    @staticmethod
    def evaluate(features: dict):
        """
        features expected:
        {
            "elbow_angle": float,
            "knee_angle": float,
            "stability_time": float
        }
        """

        if not CleanJerkRules.check_elbow_lock(features["elbow_angle"]):
            return False, "Elbow not locked"

        if not CleanJerkRules.check_knee_extension(features["knee_angle"]):
            return False, "Incomplete knee extension"

        if not CleanJerkRules.check_stability(features["stability_time"]):
            return False, "Unstable lockout"

        return True, "Valid Lift"
