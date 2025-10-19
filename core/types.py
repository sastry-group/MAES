# core/types.py
from matplotlib.colors import ListedColormap
from dataclasses import dataclass

GRID_CMAP = ListedColormap([
    "#ffffff",  # EMPTY
    "#777777",  # OBSTACLE
    "#ffeda0",  # ROI
    "#c6dbef",  # START
])

@dataclass
class GridSpec:
    width: int = 20
    height: int = 20
    neighbors: str = "4"  # "4" or "8"
    rng_seed: int = 0