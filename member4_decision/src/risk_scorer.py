"""
Risk Scorer for Member 4
Multi-factor risk assessment combining LSTM + Spatial + Density + Size
"""

from config_loader import (
    RISK_WEIGHT_LSTM,
    RISK_WEIGHT_SPATIAL,
    RISK_WEIGHT_DENSITY,
    RISK_WEIGHT_SIZE,
    RISK_CRITICAL_THRESHOLD,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD
)


class RiskScorer:
    """
    Multi-factor risk assessment

    Combines:
    - Member 3's LSTM anomaly score
    - Spatial anomaly presence (dispersal, crawling, overcrowding, imbalance)
    - Overall density level
    - Absolute crowd size

    NOTE on spatial factor weighting (updated after real-data testing):
    Individually tested against real UMN panic labels: DISPERSAL (sudden
    crowd-count drop) scored F1=0.57 alone. Crawling, overcrowding, and
    zone imbalance individually scored near-zero (0.00-0.03) -- they were
    built assuming panic means MORE crowding, but in this footage panic
    means people fleeing OUT of frame (dispersal), the opposite pattern.
    So dispersal now gets the dominant share of the spatial weight; the
    others are kept at a small share (not zero -- not fully ruled out for
    other footage/situations) rather than removed entirely.

    Output: 0-1 risk score + LOW/MEDIUM/HIGH/CRITICAL classification
    """

    def __init__(self):
        """Initialize with configured weights and thresholds"""
        self.weight_lstm = RISK_WEIGHT_LSTM
        self.weight_spatial = RISK_WEIGHT_SPATIAL
        self.weight_density = RISK_WEIGHT_DENSITY
        self.weight_size = RISK_WEIGHT_SIZE

        self.threshold_critical = RISK_CRITICAL_THRESHOLD
        self.threshold_high = RISK_HIGH_THRESHOLD
        self.threshold_medium = RISK_MEDIUM_THRESHOLD

    def compute_frame_risk(self, member3_anomaly, member4_anomalies,
                          density_level, total_count):
        """
        Compute overall risk score for a frame

        Args:
            member3_anomaly: LSTM anomaly dict from Member 3 or None
            member4_anomalies: List of spatial anomaly dicts from Member 4
            density_level: "LOW", "MEDIUM", or "HIGH"
            total_count: Total people in frame

        Returns:
            dict: {
                'risk_score': float (0.0-1.0),
                'risk_level': str (LOW/MEDIUM/HIGH/CRITICAL),
                'risk_factors': list of contributing factors
            }
        """
        risk_score = 0.0
        risk_factors = []

        # Factor 1: Member 3's anomaly score
        if member3_anomaly:
            lstm_score = member3_anomaly.get('anomaly_score', 0)
            risk_score += lstm_score * self.weight_lstm
            risk_factors.append(f"LSTM:{lstm_score:.2f}")

        # Factor 2: Member 4's spatial anomalies
        # Split the spatial weight: dispersal (proven, F1=0.57) gets the
        # dominant share; the others (near-zero individually) share the rest.
        has_dispersal = any(a['type'] == 'DISPERSAL' for a in member4_anomalies)
        has_crawling = any(a['type'] == 'CRAWLING' for a in member4_anomalies)
        has_overcrowding = any(a['type'] == 'OVERCROWDING' for a in member4_anomalies)
        has_imbalance = any(a['type'] == 'ZONE_IMBALANCE' for a in member4_anomalies)

        if has_dispersal:
            risk_score += self.weight_spatial * 0.7
            risk_factors.append("DISPERSAL")
        if has_crawling:
            risk_score += self.weight_spatial * 0.1
            risk_factors.append("CRAWLING")
        if has_overcrowding:
            risk_score += self.weight_spatial * 0.1
            risk_factors.append("OVERCROWDING")
        if has_imbalance:
            risk_score += self.weight_spatial * 0.1
            risk_factors.append("ZONE_IMBALANCE")

        # Factor 3: Density level
        density_weights = {"LOW": 0.0, "MEDIUM": 0.1, "HIGH": 0.2}
        risk_score += density_weights[density_level]
        if density_level != "LOW":
            risk_factors.append(f"DENSITY_{density_level}")

        # Factor 4: Absolute crowd size
        if total_count > 50:
            risk_score += self.weight_size
            risk_factors.append("LARGE_CROWD")

        # Clamp to [0, 1]
        risk_score = min(risk_score, 1.0)

        # Classify risk level
        if risk_score >= self.threshold_critical:
            risk_level = "CRITICAL"
        elif risk_score >= self.threshold_high:
            risk_level = "HIGH"
        elif risk_score >= self.threshold_medium:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }


# For standalone testing
if __name__ == "__main__":
    print("Testing RiskScorer...")

    scorer = RiskScorer()

    print("\nTest 1: LSTM anomaly only")
    member3_anomaly = {'anomaly_score': 0.85}
    result = scorer.compute_frame_risk(member3_anomaly, [], "LOW", 10)
    print(f"  Risk: {result['risk_score']:.2f} ({result['risk_level']})")
    print(f"  Factors: {result['risk_factors']}")

    print("\nTest 2: LSTM + Dispersal")
    member3_anomaly = {'anomaly_score': 0.75}
    member4_anomalies = [{'type': 'DISPERSAL', 'drop_z_score': 2.1}]
    result = scorer.compute_frame_risk(member3_anomaly, member4_anomalies, "LOW", 3)
    print(f"  Risk: {result['risk_score']:.2f} ({result['risk_level']})")
    print(f"  Factors: {result['risk_factors']}")

    print("\nTest 3: Low risk scenario")
    result = scorer.compute_frame_risk(None, [], "LOW", 5)
    print(f"  Risk: {result['risk_score']:.2f} ({result['risk_level']})")
    print(f"  Factors: {result['risk_factors']}")