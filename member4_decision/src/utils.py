"""
Utility functions for Member 4
Helper functions for data loading, saving, and printing
"""

import json
import os


def load_json(filepath):
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict or list: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded: {filepath}")
    return data


def save_json(data, filepath, indent=2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation (default: 2)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"✅ Saved: {filepath}")


def print_summary_stats(summary):
    """
    Print formatted summary statistics
    
    Args:
        summary: Summary dict from pipeline.generate_summary()
    """
    print("\n" + "="*60)
    print("FINAL SUMMARY - MEMBER 4 ENHANCED RESULTS")
    print("="*60)
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total videos: {summary['total_videos']}")
    print(f"   Total frames: {summary['total_frames']:,}")
    
    print(f"\n🔍 Anomaly Detection:")
    print(f"   Member 3 (LSTM): {summary['member3_anomalies']:,}")
    print(f"   Member 4 (Spatial): {summary['member4_anomalies']:,}")
    print(f"   Total anomalies: {summary['member3_anomalies'] + summary['member4_anomalies']:,}")
    
    print(f"\n🚨 Alert Summary:")
    print(f"   Total alerts: {summary['total_alerts']:,}")
    print(f"   Multi-source validations: {summary['multi_source_detections']:,} frames")
    if summary['total_frames'] > 0:
        validation_rate = (summary['multi_source_detections'] / summary['total_frames']) * 100
        print(f"   Validation rate: {validation_rate:.2f}%")
    
    print(f"\n⚠️ Risk Classification:")
    print(f"   High-risk frames: {summary['high_risk_frames']:,}")
    print(f"   Critical-risk frames: {summary['critical_risk_frames']:,}")
    print(f"   Total high+critical: {summary['high_risk_frames'] + summary['critical_risk_frames']:,}")
    
    print(f"\n📈 Alert Breakdown by Type:")
    alert_breakdown = summary['alert_system_stats']['alert_breakdown']
    for atype, count in sorted(alert_breakdown.items(), key=lambda x: x[1], reverse=True):
        if summary['total_alerts'] > 0:
            percentage = (count / summary['total_alerts']) * 100
            print(f"   {atype:20s}: {count:5,} ({percentage:5.2f}%)")
    
    print(f"\n🔄 Source Attribution:")
    source_breakdown = summary['alert_system_stats']['alerts_per_source']
    for source, count in source_breakdown.items():
        if summary['total_alerts'] > 0:
            percentage = (count / summary['total_alerts']) * 100
            print(f"   {source:10s}: {count:5,} ({percentage:5.2f}%)")
    
    print(f"\n📊 Risk Distribution:")
    risk_dist = summary['alert_system_stats']['risk_distribution']
    for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = risk_dist.get(risk_level, 0)
        if summary['total_alerts'] > 0:
            percentage = (count / summary['total_alerts']) * 100
            print(f"   {risk_level:10s}: {count:5,} ({percentage:5.2f}%)")
    
    print("\n" + "="*60 + "\n")


def print_video_summary(results):
    """
    Print summary for a single video
    
    Args:
        results: Results dict for one video from pipeline.process_video()
    """
    print(f"\n{'='*60}")
    print(f"Video {results['video_id']} Summary")
    print(f"{'='*60}")
    print(f"   Total frames: {results['total_frames']}")
    print(f"   Member 3 anomalies: {results['member3_anomalies']}")
    print(f"   Member 4 anomalies: {results['member4_anomalies']}")
    print(f"   Total alerts: {results['total_alerts']}")
    print(f"   Multi-source detections: {results['multi_source_detections']}")
    print(f"   High-risk frames: {results['high_risk_frames']}")
    print(f"   Critical-risk frames: {results['critical_risk_frames']}")
    print(f"{'='*60}\n")


def calculate_detection_rates(summary):
    """
    Calculate detection rates as percentages
    
    Args:
        summary: Summary dict from pipeline.generate_summary()
    
    Returns:
        dict: Detection rates for each anomaly type
    """
    rates = {}
    total_frames = summary['total_frames']
    
    if total_frames > 0:
        alert_breakdown = summary['alert_system_stats']['alert_breakdown']
        for atype, count in alert_breakdown.items():
            rates[atype] = (count / total_frames) * 100
    
    return rates


def print_detection_rates(summary):
    """
    Print detection rates
    
    Args:
        summary: Summary dict from pipeline.generate_summary()
    """
    rates = calculate_detection_rates(summary)
    
    print("\n" + "="*60)
    print("DETECTION RATES")
    print("="*60)
    for atype, rate in sorted(rates.items(), key=lambda x: x[1], reverse=True):
        print(f"   {atype:20s}: {rate:6.2f}% of frames")
    print("="*60 + "\n")


def validate_data_integrity(member1_data, member3_anomalies):
    """
    Validate data integrity between members
    
    Args:
        member1_data: Member 1's detection data
        member3_anomalies: Member 3's anomaly data
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    print("\n" + "="*60)
    print("VALIDATING DATA INTEGRITY")
    print("="*60)
    
    m1_videos = set(member1_data.keys())
    m3_videos = set(member3_anomalies.keys())
    
    print(f"   Member 1 videos: {len(m1_videos)}")
    print(f"   Member 3 videos: {len(m3_videos)}")
    
    # Check for missing videos
    missing_in_m3 = m1_videos - m3_videos
    missing_in_m1 = m3_videos - m1_videos
    
    if missing_in_m3:
        print(f"\n⚠️  Videos in M1 but not M3: {sorted(missing_in_m3)}")
    
    if missing_in_m1:
        print(f"\n⚠️  Videos in M3 but not M1: {sorted(missing_in_m1)}")
    
    common_videos = m1_videos & m3_videos
    print(f"\n✅ Common videos: {len(common_videos)}")
    
    # Check frame counts match
    mismatches = []
    for vid in common_videos:
        m1_frames = len(member1_data[vid]['detections'])
        m3_frames = member3_anomalies[vid].get('total_frames', 0)
        
        if m1_frames != m3_frames:
            mismatches.append((vid, m1_frames, m3_frames))
    
    if mismatches:
        print(f"\n⚠️  Frame count mismatches:")
        for vid, m1_f, m3_f in mismatches:
            print(f"   Video {vid}: M1={m1_f}, M3={m3_f}")
    else:
        print(f"\n✅ All frame counts match")
    
    print("="*60 + "\n")
    
    return len(common_videos) > 0 and len(mismatches) == 0


# For standalone testing
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test JSON operations
    test_data = {'test': 'data', 'count': 123}
    save_json(test_data, 'test_output.json')
    loaded = load_json('test_output.json')
    print(f"Loaded data: {loaded}")
    
    # Test detection rates
    mock_summary = {
        'total_frames': 1000,
        'total_alerts': 150,
        'alert_system_stats': {
            'alert_breakdown': {
                'CRAWLING': 100,
                'OVERCROWDING': 30,
                'LSTM_ANOMALY': 20
            }
        }
    }
    
    print_detection_rates(mock_summary)
    
    # Clean up
    if os.path.exists('test_output.json'):
        os.remove('test_output.json')
        print("Cleaned up test file")