"""
Enhanced Density Estimator for Member 4
Grid-based crowd density analysis with zone integration
"""

import numpy as np
from config_loader import (
    GRID_SIZE, GRID_ROWS, GRID_COLS, FRAME_WIDTH, FRAME_HEIGHT,
    ZONE_MAPPING, DENSITY_LOW_THRESHOLD, DENSITY_MEDIUM_THRESHOLD,
    HOTSPOT_THRESHOLD
)


class EnhancedDensityEstimator:
    """
    Grid-based density estimation with zone awareness
    
    Features:
    - 4×3 grid (12 zones) for spatial analysis
    - Integration with Member 2's LEFT/CENTER/RIGHT zones
    - Hotspot detection with zone tagging
    """
    
    def __init__(self, grid_size=GRID_SIZE, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
        """
        Initialize density estimator
        
        Args:
            grid_size: Cell size in pixels (default: 160)
            frame_width: Frame width (default: 640)
            frame_height: Frame height (default: 360)
        """
        self.grid_size = grid_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows = GRID_ROWS
        self.grid_cols = GRID_COLS
        self.zone_mapping = ZONE_MAPPING
        
    def compute_density_grid(self, detections):
        """
        Compute people count per grid cell
        
        Args:
            detections: List of [x1, y1, x2, y2, conf]
        
        Returns:
            numpy.ndarray: 2D array (3×4) with people count per cell
        """
        density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            
            # Center point of bounding box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Find grid cell
            col = int(cx // self.grid_size)
            row = int(cy // (self.frame_height / self.grid_rows))
            
            # Bounds checking
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                density_grid[row, col] += 1
        
        return density_grid
    
    def get_zone_densities(self, density_grid):
        """
        Aggregate grid cells into Member 2's zone structure
        
        Args:
            density_grid: 2D numpy array from compute_density_grid()
        
        Returns:
            dict: {'LEFT': count, 'CENTER': count, 'RIGHT': count}
        """
        zone_densities = {'LEFT': 0, 'CENTER': 0, 'RIGHT': 0}
        
        for col in range(self.grid_cols):
            zone = self.zone_mapping[col]
            zone_densities[zone] += int(density_grid[:, col].sum())
        
        return zone_densities
    
    def classify_density(self, total_count):
        """
        Classify overall frame density level
        
        Args:
            total_count: Total number of people in frame
        
        Returns:
            tuple: (density_level, color) e.g., ("MEDIUM", (0, 165, 255))
        """
        if total_count < DENSITY_LOW_THRESHOLD:
            return "LOW", (0, 255, 0)  # Green
        elif total_count < DENSITY_MEDIUM_THRESHOLD:
            return "MEDIUM", (0, 165, 255)  # Orange
        else:
            return "HIGH", (0, 0, 255)  # Red
    
    def get_hotspots(self, density_grid, threshold=HOTSPOT_THRESHOLD):
        """
        Identify overcrowded cells with zone information
        
        Args:
            density_grid: 2D numpy array
            threshold: Minimum people count to be considered hotspot
        
        Returns:
            list: List of hotspot dictionaries with position, count, zone
        """
        hotspots = []
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if density_grid[r, c] >= threshold:
                    hotspots.append({
                        'grid_pos': (r, c),
                        'pixel_pos': (
                            c * self.grid_size,
                            r * (self.frame_height // self.grid_rows)
                        ),
                        'count': int(density_grid[r, c]),
                        'zone': self.zone_mapping[c]
                    })
        
        return hotspots


# For standalone testing
if __name__ == "__main__":
    print("Testing EnhancedDensityEstimator...")
    
    # Sample detections (x1, y1, x2, y2, conf)
    sample_detections = [
        [100, 150, 150, 250, 0.9],
        [200, 160, 250, 260, 0.85],
        [500, 180, 550, 280, 0.92]
    ]
    
    estimator = EnhancedDensityEstimator()
    
    # Test density grid
    grid = estimator.compute_density_grid(sample_detections)
    print(f"Density Grid:\n{grid}")
    
    # Test zone densities
    zones = estimator.get_zone_densities(grid)
    print(f"\nZone Densities: {zones}")
    
    # Test classification
    level, color = estimator.classify_density(len(sample_detections))
    print(f"\nDensity Level: {level} (color: {color})")
    
    # Test hotspots
    hotspots = estimator.get_hotspots(grid)
    print(f"\nHotspots: {len(hotspots)} found")