"""
Visualization functions for Member 4
Creates comprehensive charts and heatmaps
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_comprehensive_visualization(summary, all_results, output_path):
    """
    Create 4-panel comprehensive analysis chart
    
    Args:
        summary: Summary dict from pipeline.generate_summary()
        all_results: All results dict from pipeline.process_all_videos()
        output_path: Path to save PNG file
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Member 4 Enhanced Analysis - Review 2', 
                fontsize=20, fontweight='bold', y=0.995)
    
    colors_main = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    colors_risk = {
        'CRITICAL': '#8B0000',
        'HIGH': '#FF6B6B',
        'MEDIUM': '#FFE66D',
        'LOW': '#95E1D3'
    }
    
    # 1. Alert Distribution by Type (Top Left)
    ax1 = axes[0, 0]
    alert_types = summary['alert_system_stats']['alert_breakdown']
    sorted_types = sorted(alert_types.items(), key=lambda x: x[1], reverse=True)
    types, counts = zip(*sorted_types) if sorted_types else ([], [])
    
    if types:
        bars1 = ax1.barh(types, counts, color=colors_main[:len(types)])
        ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Alert Distribution by Type', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        for i, (bar, count) in enumerate(zip(bars1, counts)):
            ax1.text(count, i, f' {count:,}', va='center', fontsize=10, fontweight='bold')
    
    # 2. Alerts per Video (Top Right)
    ax2 = axes[0, 1]
    video_ids = sorted(all_results.keys())
    alert_counts = [all_results[vid]['total_alerts'] for vid in video_ids]
    bars2 = ax2.bar(video_ids, alert_counts, color='#4ECDC4', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Video ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Alerts', fontsize=12, fontweight='bold')
    ax2.set_title('Alerts per Video', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight top 3
    if len(alert_counts) >= 3:
        top_3_indices = sorted(range(len(alert_counts)), key=lambda i: alert_counts[i], reverse=True)[:3]
        for idx in top_3_indices:
            bars2[idx].set_color('#FF6B6B')
    
    # 3. Source Attribution (Bottom Left)
    ax3 = axes[1, 0]
    source_data = summary['alert_system_stats']['alerts_per_source']
    if source_data:
        labels = list(source_data.keys())
        sizes = list(source_data.values())
        colors_pie = ['#95E1D3', '#F38181']
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors_pie[:len(labels)], startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
    ax3.set_title('Alert Source Distribution', fontsize=14, fontweight='bold', pad=15)
    
    # 4. Risk Level Distribution (Bottom Right)
    ax4 = axes[1, 1]
    risk_levels_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    risk_counts = [
        summary['alert_system_stats']['risk_distribution'].get(level, 0)
        for level in risk_levels_order
    ]
    bars4 = ax4.bar(
        risk_levels_order, risk_counts,
        color=[colors_risk[level] for level in risk_levels_order],
        edgecolor='black', linewidth=1.5
    )
    ax4.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Alert Count', fontsize=12, fontweight='bold')
    ax4.set_title('Risk Level Distribution', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars4, risk_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_zone_heatmap(all_results, output_path):
    """
    Create zone-wise crowd distribution chart
    
    Args:
        all_results: All results dict from pipeline.process_all_videos()
        output_path: Path to save PNG file
    """
    # Aggregate zone densities
    zone_totals = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
    frame_count = 0
    
    for video_results in all_results.values():
        for frame_data in video_results['frames']:
            for zone, count in frame_data['zone_densities'].items():
                zone_totals[zone] += count
            frame_count += 1
    
    # Calculate averages
    zone_averages = {
        zone: total / frame_count if frame_count > 0 else 0
        for zone, total in zone_totals.items()
    }
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Zone-Wise Crowd Distribution (Average)', 
                fontsize=18, fontweight='bold')
    
    zones = ['LEFT', 'CENTER', 'RIGHT']
    averages = [zone_averages[z] for z in zones]
    colors_zones = ['#3498DB', '#E74C3C', '#2ECC71']
    
    bars = ax.bar(zones, averages, color=colors_zones, edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_ylabel('Average People Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Zone', fontsize=14, fontweight='bold')
    ax.set_title(f'Data from {frame_count:,} frames across {len(all_results)} videos',
                fontsize=12, pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(averages) * 1.2 if averages else 1)
    
    # Add value labels
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{avg:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add zone descriptions
    zone_desc = {
        'LEFT': 'Columns 0-1\n(0-320px)',
        'CENTER': 'Column 2\n(320-480px)',
        'RIGHT': 'Column 3\n(480-640px)'
    }
    max_avg = max(averages) if averages else 1
    for i, zone in enumerate(zones):
        ax.text(i, -max_avg*0.15, zone_desc[zone],
               ha='center', va='top', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


# For standalone testing
if __name__ == "__main__":
    print("Testing visualization functions...")
    
    # Mock data
    mock_summary = {
        'total_videos': 2,
        'total_frames': 100,
        'total_alerts': 50,
        'alert_system_stats': {
            'alert_breakdown': {
                'CRAWLING': 30,
                'OVERCROWDING': 10,
                'LSTM_ANOMALY': 10
            },
            'alerts_per_source': {
                'MEMBER3': 10,
                'MEMBER4': 40
            },
            'risk_distribution': {
                'LOW': 10,
                'MEDIUM': 20,
                'HIGH': 15,
                'CRITICAL': 5
            }
        }
    }
    
    mock_results = {
        "01": {
            'total_alerts': 30,
            'frames': [
                {'zone_densities': {'LEFT': 5, 'CENTER': 3, 'RIGHT': 2}},
                {'zone_densities': {'LEFT': 4, 'CENTER': 4, 'RIGHT': 2}}
            ]
        },
        "02": {
            'total_alerts': 20,
            'frames': [
                {'zone_densities': {'LEFT': 6, 'CENTER': 2, 'RIGHT': 3}}
            ]
        }
    }
    
    # Test visualizations
    create_comprehensive_visualization(mock_summary, mock_results, 'test_analysis.png')
    create_zone_heatmap(mock_results, 'test_heatmap.png')
    
    print("Test visualizations created!")
