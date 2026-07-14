from config import RISK_WEIGHTS, RISK_LEVELS, DENSITY_THRESHOLDS


class RiskScorer:
    def __init__(self, weights=RISK_WEIGHTS):
        self.weights = weights

    def compute_density_score(self, count):
        if count < DENSITY_THRESHOLDS['sparse'][1]:
            return count / DENSITY_THRESHOLDS['sparse'][1] * 0.33
        elif count < DENSITY_THRESHOLDS['medium'][1]:
            return 0.33 + (count - 15) / 25 * 0.33
        else:
            return 0.66 + min((count - 40) / 60, 1.0) * 0.34

    def compute_size_score(self, count):
        return min(count / 50.0, 1.0)

    def compute_risk_score(self, lstm_score, spatial_score, person_count):
        lstm_norm    = min(max(lstm_score, 0.0), 1.0)
        density_norm = self.compute_density_score(person_count)
        size_norm    = self.compute_size_score(person_count)

        risk_score = (
            self.weights['lstm']    * lstm_norm    +
            self.weights['spatial'] * spatial_score +
            self.weights['density'] * density_norm  +
            self.weights['size']    * size_norm
        )

        components = {
            'lstm': lstm_norm, 'spatial': spatial_score,
            'density': density_norm, 'size': size_norm,
            'weighted_lstm':    self.weights['lstm']    * lstm_norm,
            'weighted_spatial': self.weights['spatial'] * spatial_score,
            'weighted_density': self.weights['density'] * density_norm,
            'weighted_size':    self.weights['size']    * size_norm,
        }
        return risk_score, components

    def classify_risk_level(self, risk_score):
        for level, (lo, hi) in RISK_LEVELS.items():
            if lo <= risk_score < hi:
                return level
        return 'CRITICAL'