from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class RoutingState:
    """
    MINIMAL state for ML consumption
    NO full-grid dumps, NO large dictionaries

    Features:
    - 13D patch encoding (5×5 window)
    - Enhanced global context
    - Candidate neighbors
    """
    # Current search position
    current_pos: Tuple[int, int]  # (y, x)
    goal_pos: Tuple[int, int]     # (y, x)
    start_pos: Tuple[int, int]    # (y, x)

    # Path costs
    g_cost: float  # Cost to reach current
    h_cost: float  # Manhattan heuristic to goal (for logging/training)

    # Local neighborhood (5×5 window centered on current)
    # Features per patch: 13 dimensions
    local_window: np.ndarray  # Shape (5, 5, 13)

    # Basic global context
    grid_shape: Tuple[int, int]
    current_cycle: int
    current_stv: int
    n_active_gates: int
    n_waiting_gates: int
    gate_id: str  # Which gate we're routing

    # Enhanced global features
    active_area_density: float    # (# active patches) / (H * W)
    route_congestion: float       # # locked patches in 7×7 region
    factory_load: int             # # active extern/factory operations
    global_distance_ratio: float  # manhattan(start,end) / max(H,W)

    # Candidate neighbors for current expansion
    neighbors: List[Tuple[int, int]]  # [(y1,x1), (y2,x2), ...]
    neighbor_costs: List[float]       # [g1, g2, ...]
