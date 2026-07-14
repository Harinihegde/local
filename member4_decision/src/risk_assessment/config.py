FRAME_WIDTH  = 360
FRAME_HEIGHT = 240
GRID_ROWS = 4
GRID_COLS = 3
CELL_HEIGHT = FRAME_HEIGHT // GRID_ROWS
CELL_WIDTH  = FRAME_WIDTH  // GRID_COLS

THRESHOLDS = {
    'crawling_height':   75,
    'overcrowding_count': 6,
    'zone_imbalance':   0.70,
    'lstm_anomaly':     0.5,
}

RISK_WEIGHTS = {
    'lstm':    0.40,
    'spatial': 0.30,
    'density': 0.20,
    'size':    0.10
}

RISK_LEVELS = {
    'LOW':      (0.0,  0.25),
    'MEDIUM':   (0.25, 0.50),
    'HIGH':     (0.50, 0.75),
    'CRITICAL': (0.75, 1.0)
}

DENSITY_THRESHOLDS = {
    'sparse': (0,  15),
    'medium': (15, 40),
    'dense':  (40, 1000)
}