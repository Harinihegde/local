"""
Enhanced Alert System for Member 4
Unified alert management with source tracking and multi-member validation
"""

import json
from datetime import datetime
from collections import defaultdict


class EnhancedAlertSystem:
    """
    Alert management with source attribution
    
    Features:
    - Track alerts from Member 3 (LSTM) and Member 4 (Spatial)
    - Multi-source validation (frames flagged by both members)
    - Risk score integration
    - JSON export capability
    """
    
    def __init__(self):
        """Initialize alert system"""
        self.alert_log = []
        self.alert_counts = defaultdict(int)
        self.source_counts = defaultdict(int)
        self.multi_source_frames = set()
        
    def trigger_alert(self, video_id, frame_num, anomaly, source_member, risk_info=None):
        """
        Create and log an alert
        
        Args:
            video_id: Video identifier (e.g., "01", "02")
            frame_num: Frame number
            anomaly: Anomaly dictionary with 'type', 'severity', etc.
            source_member: "MEMBER3" or "MEMBER4"
            risk_info: Optional risk scoring dict from RiskScorer
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'video_id': video_id,
            'frame': frame_num,
            'anomaly_type': anomaly.get('type', 'LSTM_ANOMALY'),
            'severity': anomaly.get('severity', 'MEDIUM'),
            'source_member': source_member,
            'zone': anomaly.get('zone', anomaly.get('dominant_zone', 'UNKNOWN')),
            'details': anomaly
        }
        
        # Add risk information if provided
        if risk_info:
            alert['risk_score'] = risk_info['risk_score']
            alert['risk_level'] = risk_info['risk_level']
            alert['risk_factors'] = risk_info['risk_factors']
        
        self.alert_log.append(alert)
        
        # Update statistics
        self.alert_counts[alert['anomaly_type']] += 1
        self.source_counts[source_member] += 1
        
        # Track multi-source frames for validation
        frame_key = f"{video_id}_{frame_num}"
        self.multi_source_frames.add(frame_key)
    
    def get_summary(self):
        """
        Get comprehensive statistics
        
        Returns:
            dict: Summary statistics including counts, breakdown, distribution
        """
        return {
            'total_alerts': len(self.alert_log),
            'alert_breakdown': dict(self.alert_counts),
            'alerts_per_source': dict(self.source_counts),
            'multi_source_frames': len(self.multi_source_frames),
            'risk_distribution': self._count_by_risk_level()
        }
    
    def _count_by_risk_level(self):
        """Count alerts by risk level"""
        counts = defaultdict(int)
        for alert in self.alert_log:
            if 'risk_level' in alert:
                counts[alert['risk_level']] += 1
        return dict(counts)
    
    def save_alert_log(self, output_path):
        """
        Save alert log to JSON file
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.alert_log, f, indent=2)
        print(f"✅ Saved {len(self.alert_log)} alerts to {output_path}")
    
    def get_alerts_by_video(self, video_id):
        """
        Get all alerts for a specific video
        
        Args:
            video_id: Video identifier
        
        Returns:
            list: Alerts for the specified video
        """
        return [alert for alert in self.alert_log if alert['video_id'] == video_id]
    
    def get_high_priority_alerts(self, min_risk_level='HIGH'):
        """
        Get high-priority alerts (HIGH or CRITICAL risk)
        
        Args:
            min_risk_level: Minimum risk level to include ('HIGH' or 'CRITICAL')
        
        Returns:
            list: High-priority alerts
        """
        priority_levels = {'CRITICAL', 'HIGH'} if min_risk_level == 'HIGH' else {'CRITICAL'}
        return [
            alert for alert in self.alert_log
            if alert.get('risk_level') in priority_levels
        ]
    
    def print_summary(self):
        """Print formatted summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("ALERT SYSTEM SUMMARY")
        print("="*60)
        
        print(f"\n📊 Total alerts: {summary['total_alerts']:,}")
        print(f"   Multi-source validations: {summary['multi_source_frames']:,}")
        
        print(f"\n🔍 Alert breakdown by type:")
        for atype, count in sorted(summary['alert_breakdown'].items(), 
                                   key=lambda x: x[1], reverse=True):
            pct = (count / summary['total_alerts']) * 100 if summary['total_alerts'] > 0 else 0
            print(f"   {atype:20s}: {count:5,} ({pct:5.2f}%)")
        
        print(f"\n👥 Alerts per source:")
        for source, count in summary['alerts_per_source'].items():
            pct = (count / summary['total_alerts']) * 100 if summary['total_alerts'] > 0 else 0
            print(f"   {source:10s}: {count:5,} ({pct:5.2f}%)")
        
        print(f"\n⚠️ Risk distribution:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = summary['risk_distribution'].get(level, 0)
            pct = (count / summary['total_alerts']) * 100 if summary['total_alerts'] > 0 else 0
            print(f"   {level:10s}: {count:5,} ({pct:5.2f}%)")
        
        print("="*60 + "\n")


# For standalone testing
if __name__ == "__main__":
    print("Testing EnhancedAlertSystem...")
    
    system = EnhancedAlertSystem()
    
    # Test alert creation
    print("\nCreating sample alerts...")
    
    # Alert 1: LSTM anomaly
    system.trigger_alert(
        "01", 120,
        {'type': 'LSTM_ANOMALY', 'anomaly_score': 0.85},
        'MEMBER3',
        {'risk_score': 0.85, 'risk_level': 'HIGH', 'risk_factors': ['LSTM:0.85']}
    )
    
    # Alert 2: Crawling
    system.trigger_alert(
        "01", 120,
        {'type': 'CRAWLING', 'height': 45, 'zone': 'CENTER', 'severity': 'HIGH'},
        'MEMBER4',
        {'risk_score': 0.92, 'risk_level': 'CRITICAL', 'risk_factors': ['LSTM:0.85', 'CRAWLING']}
    )
    
    # Alert 3: Overcrowding
    system.trigger_alert(
        "02", 200,
        {'type': 'OVERCROWDING', 'count': 12, 'zone': 'LEFT', 'severity': 'CRITICAL'},
        'MEMBER4',
        {'risk_score': 0.65, 'risk_level': 'HIGH', 'risk_factors': ['OVERCROWDING', 'DENSITY_HIGH']}
    )
    
    # Print summary
    system.print_summary()
    
    # Test filtering
    high_priority = system.get_high_priority_alerts()
    print(f"High priority alerts: {len(high_priority)}")
    
    # Test video filtering
    video_01_alerts = system.get_alerts_by_video("01")
    print(f"Video 01 alerts: {len(video_01_alerts)}")