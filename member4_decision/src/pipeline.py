"""
Member 4 Pipeline - Complete Integration
Combines all Member 4 components into unified processing pipeline
"""

from density_estimator import EnhancedDensityEstimator
from anomaly_detector import EnhancedAnomalyDetector
from member4_decision.src.dispersal_detector import DispersalDetector
from risk_scorer import RiskScorer
from alert_system import EnhancedAlertSystem
from member3_to4 import zscore_to_probability


# Member 3's decision cutoff (the boundary between normal/anomaly it
# tuned during its own training). Used here to convert its raw z-score
# into a clean 0-to-1 number Member 4 can safely combine with other signals.
# NOTE: this is currently a single fixed placeholder value (~5.0-6.0, the
# rough range Member 3's per-video tuning landed on). For real deployment,
# Member 3 should settle on and report ONE final chosen cutoff (from
# training on all available data), and that exact number should replace
# this placeholder.
MEMBER3_Z_CUTOFF = 5.5


class Member4Pipeline:
    """
    Complete Member 4 integration pipeline

    Integrates:
    - Member 1's fine-tuned detections (bounding boxes)
    - Member 2's zonal features (optional, for validation)
    - Member 3's LSTM temporal anomalies
    - Member 4's spatial analysis + risk scoring

    Output: Unified alert system with risk prioritization
    """

    def __init__(self, member1_data, member2_features_folder, member3_anomalies):
        """
        Initialize pipeline with all members' data

        Args:
            member1_data: Dict with detection data from Member 1
            member2_features_folder: Path to Member 2's feature CSVs
            member3_anomalies: Dict with LSTM anomalies from Member 3
        """
        self.member1_detections = member1_data
        self.member2_features_folder = member2_features_folder
        self.member3_anomalies = member3_anomalies

        # Initialize Member 4 components
        self.density_est = EnhancedDensityEstimator()
        self.anomaly_detector = EnhancedAnomalyDetector()
        self.risk_scorer = RiskScorer()
        self.alert_system = EnhancedAlertSystem()

        print("\n" + "="*60)
        print("✅ Member 4 Pipeline Initialized")
        print("="*60)
        print(f"   Videos to process: {len(member1_data)}")
        print(f"   Components: Density, Anomaly, Risk, Alert")

    def process_video(self, video_id):
        """
        Process one video with complete analysis

        Args:
            video_id: Video identifier (e.g., "01", "02")

        Returns:
            dict: Video processing results with frame-by-frame data
        """
        print(f"\n{'='*60}")
        print(f"Processing Video: {video_id}")
        print(f"{'='*60}")

        # Load Member 1's detections
        video_detections = self.member1_detections[video_id]['detections']

        # Load Member 3's LSTM anomalies
        member3_data = self.member3_anomalies.get(video_id, {})
        member3_anomaly_list = member3_data.get('anomalies', [])

        # Group Member 3's anomalies by frame
        anomalies_by_frame = {}
        for anomaly in member3_anomaly_list:
            frame_num = anomaly.get('frame')
            anomalies_by_frame[frame_num] = anomaly

        # Fresh dispersal detector for THIS video only — it needs its own
        # rolling history, not shared across different videos.
        dispersal_detector = DispersalDetector(z_threshold=0.7)

        # Initialize results
        results = {
            'video_id': video_id,
            'total_frames': len(video_detections),
            'frames': [],
            'member3_anomalies': len(member3_anomaly_list),
            'member4_anomalies': 0,
            'total_alerts': 0,
            'high_risk_frames': 0,
            'critical_risk_frames': 0,
            'multi_source_detections': 0
        }

        # Process each frame
        for frame_num, frame_dets in enumerate(video_detections):
            # 1. Density analysis
            density_grid = self.density_est.compute_density_grid(frame_dets)
            zone_densities = self.density_est.get_zone_densities(density_grid)
            density_level, _ = self.density_est.classify_density(len(frame_dets))
            hotspots = self.density_est.get_hotspots(density_grid)

            # 2. Member 4's spatial anomalies
            crawling = self.anomaly_detector.detect_crawling(frame_dets)
            overcrowding = self.anomaly_detector.detect_overcrowding(
                density_grid, zone_densities
            )
            zone_imbalance = self.anomaly_detector.detect_zone_imbalance(
                zone_densities, len(frame_dets)
            )

            member4_anomalies = crawling + overcrowding
            if zone_imbalance:
                member4_anomalies.append(zone_imbalance)

            # Sudden crowd-count drop check — confirmed the strongest
            # working spatial signal (F1=0.57 alone vs. near-zero for the
            # older count/crowding-based checks, which assumed panic means
            # MORE people, when in this footage it means people fleeing
            # out of frame).
            is_dispersal, dispersal_z = dispersal_detector.update(len(frame_dets))
            if is_dispersal:
                member4_anomalies.append({
                    'type': 'DISPERSAL',
                    'drop_z_score': dispersal_z,
                    'severity': 'HIGH' if dispersal_z > 1.5 else 'MEDIUM',
                })

            results['member4_anomalies'] += len(member4_anomalies)

            # 3. Member 3's LSTM anomaly for this frame
            #    IMPORTANT: Member 3 now reports a raw z-score (can be any
            #    size, e.g. 2, 8, 15 — not a 0-to-1 number). Convert it here
            #    before RiskScorer sees it, so its weighted-sum math (capped
            #    at 1.0 total) stays meaningful instead of saturating instantly.
            member3_anomaly = anomalies_by_frame.get(frame_num)
            if member3_anomaly is not None:
                raw_z = member3_anomaly.get('anomaly_score', 0)
                probability = zscore_to_probability(raw_z, cutoff=MEMBER3_Z_CUTOFF)
                # Keep the raw z-score too (renamed) for debugging/inspection,
                # but 'anomaly_score' — the field RiskScorer actually reads —
                # is now the translated 0-to-1 number.
                member3_anomaly = {
                    **member3_anomaly,
                    'anomaly_score_raw_z': raw_z,
                    'anomaly_score': probability,
                }
                print(f"  Frame {frame_num}: Member 3 raw z-score = {raw_z} "
                      f"-> translated to {probability:.3f} (0-to-1 scale)")

            # 4. Risk scoring
            risk_info = self.risk_scorer.compute_frame_risk(
                member3_anomaly,
                member4_anomalies,
                density_level,
                len(frame_dets)
            )

            # Track high-risk frames
            if risk_info['risk_level'] == 'HIGH':
                results['high_risk_frames'] += 1
            elif risk_info['risk_level'] == 'CRITICAL':
                results['critical_risk_frames'] += 1

            # Track multi-source detections
            if member3_anomaly and member4_anomalies:
                results['multi_source_detections'] += 1

            # 5. Trigger alerts
            if member3_anomaly:
                self.alert_system.trigger_alert(
                    video_id, frame_num, member3_anomaly,
                    source_member='MEMBER3',
                    risk_info=risk_info
                )
                results['total_alerts'] += 1

            for anomaly in member4_anomalies:
                self.alert_system.trigger_alert(
                    video_id, frame_num, anomaly,
                    source_member='MEMBER4',
                    risk_info=risk_info
                )
                results['total_alerts'] += 1

            # 6. Store frame results
            results['frames'].append({
                'frame_num': frame_num,
                'people_count': len(frame_dets),
                'density_level': density_level,
                'zone_densities': zone_densities,
                'hotspots': len(hotspots),
                'member3_anomaly': member3_anomaly is not None,
                'member4_anomalies': len(member4_anomalies),
                'risk_score': risk_info['risk_score'],
                'risk_level': risk_info['risk_level']
            })

        print(f"✅ Processed {len(video_detections)} frames")
        print(f"   Member 3 (LSTM): {results['member3_anomalies']}")
        print(f"   Member 4 (Spatial): {results['member4_anomalies']}")
        print(f"   Multi-source: {results['multi_source_detections']}")
        print(f"   Total alerts: {results['total_alerts']}")

        return results

    def process_all_videos(self):
        """
        Process all videos in dataset

        Returns:
            dict: Results for all videos
        """
        all_results = {}

        print("\n" + "="*60)
        print("PROCESSING ALL VIDEOS")
        print("="*60)

        for video_id in sorted(self.member1_detections.keys()):
            results = self.process_video(video_id)
            all_results[video_id] = results

        return all_results

    def generate_summary(self, all_results):
        """
        Generate comprehensive summary statistics

        Args:
            all_results: Dict from process_all_videos()

        Returns:
            dict: Overall summary across all videos
        """
        summary = {
            'total_videos': len(all_results),
            'total_frames': sum(r['total_frames'] for r in all_results.values()),
            'member3_anomalies': sum(r['member3_anomalies'] for r in all_results.values()),
            'member4_anomalies': sum(r['member4_anomalies'] for r in all_results.values()),
            'total_alerts': sum(r['total_alerts'] for r in all_results.values()),
            'high_risk_frames': sum(r['high_risk_frames'] for r in all_results.values()),
            'critical_risk_frames': sum(r['critical_risk_frames'] for r in all_results.values()),
            'multi_source_detections': sum(r['multi_source_detections'] for r in all_results.values()),
            'alert_system_stats': self.alert_system.get_summary()
        }

        return summary


# For standalone testing
if __name__ == "__main__":
    import json

    print("Testing Member4Pipeline...")

    # Mock data for testing
    mock_member1 = {
        "01": {
            "detections": [
                [[100, 150, 150, 250, 0.9], [200, 160, 250, 260, 0.85]],
                [[105, 155, 155, 255, 0.88]]
            ],
            "fps": 25.0
        }
    }

    # NOTE: anomaly_score here is now a raw z-score (e.g. 8.5), not 0-1,
    # matching what Member 3's fixed pipeline actually outputs.
    mock_member3 = {
        "01": {
            "anomalies": [
                {"frame": 0, "anomaly_score": 8.5}
            ]
        }
    }

    # Create pipeline
    pipeline = Member4Pipeline(mock_member1, "./features", mock_member3)

    # Process
    results = pipeline.process_video("01")
    print(f"\nResults for video 01:")
    print(f"  Total frames: {results['total_frames']}")
    print(f"  Total alerts: {results['total_alerts']}")

    # Generate summary
    all_results = {"01": results}
    summary = pipeline.generate_summary(all_results)
    print(f"\nOverall summary:")
    print(f"  Total alerts: {summary['total_alerts']}")