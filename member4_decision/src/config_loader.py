"""
Configuration loader for Member 4
Loads settings from config.yaml
"""

import yaml
import os

def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    # Get project root (2 levels up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    full_path = os.path.join(project_root, 'member4_decision', config_path)

    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

# Load config
CONFIG = load_config()

# Export commonly used values
FRAME_WIDTH = CONFIG['frame']['width']
FRAME_HEIGHT = CONFIG['frame']['height']
GRID_SIZE = CONFIG['grid']['size']
GRID_ROWS = CONFIG['grid']['rows']
GRID_COLS = CONFIG['grid']['cols']

CRAWLING_HEIGHT_THRESHOLD = CONFIG['thresholds']['crawling_height']
OVERCROWDING_THRESHOLD = CONFIG['thresholds']['overcrowding']
ZONE_IMBALANCE_RATIO = CONFIG['thresholds']['zone_imbalance']
HOTSPOT_THRESHOLD = CONFIG['thresholds']['hotspot']

DENSITY_LOW_THRESHOLD = CONFIG['density']['low']
DENSITY_MEDIUM_THRESHOLD = CONFIG['density']['medium']

RISK_WEIGHT_LSTM = CONFIG['risk_weights']['lstm']
RISK_WEIGHT_SPATIAL = CONFIG['risk_weights']['spatial']
RISK_WEIGHT_DENSITY = CONFIG['risk_weights']['density']
RISK_WEIGHT_SIZE = CONFIG['risk_weights']['size']

RISK_CRITICAL_THRESHOLD = CONFIG['risk_levels']['critical']
RISK_HIGH_THRESHOLD = CONFIG['risk_levels']['high']
RISK_MEDIUM_THRESHOLD = CONFIG['risk_levels']['medium']

ZONE_MAPPING = CONFIG['zones']['mapping']

# FIX: these boundaries used to be hardcoded (213 / 426) directly below,
# completely ignoring whatever was written in config.yaml's zones.boundaries
# section. That meant editing the config file had NO EFFECT on this function
# at all — it silently kept using numbers correct only for a 640-wide frame.
# Now it actually reads the real boundaries from config, so fixing the YAML
# (e.g. for UMN's 320-wide frames) actually changes this function's behavior.
_ZONE_BOUNDARIES = CONFIG['zones']['boundaries']
_LEFT_MAX = _ZONE_BOUNDARIES['left'][1]
_CENTER_MAX = _ZONE_BOUNDARIES['center'][1]

def get_zone_from_x_coordinate(x):
    """Determine zone from x-coordinate, using boundaries from config.yaml
    (not hardcoded — see fix note above)."""
    if x < _LEFT_MAX:
        return 'LEFT'
    elif x < _CENTER_MAX:
        return 'CENTER'
    else:
        return 'RIGHT'