import json
from typing import Tuple


class JsonlLogger:
    """
    Logs A* search expansions to JSONL for training data

    Minimal logging: state + action per expansion
    """

    def __init__(self, output_path: str, enabled: bool = True):
        self.output_path = output_path
        self.enabled = enabled
        self.file = None
        self.search_id = 0
        self.expansion_count = 0

    def __enter__(self):
        if self.enabled:
            self.file = open(self.output_path, 'a')
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def log_expansion(self,
                      state: 'RoutingState',
                      chosen_neighbor: Tuple[int, int],
                      chosen_cost: float):
        """
        Log ONE expansion (called ONCE per iteration)
        """
        if not self.enabled:
            return

        record = {
            "type": "expansion",
            "search_id": self.search_id,
            "expansion_id": self.expansion_count,

            # Minimal state
            "current": list(state.current_pos),
            "goal": list(state.goal_pos),
            "start": list(state.start_pos),
            "g_cost": state.g_cost,
            "h_cost": state.h_cost,

            # Local features (compact representation)
            "local_window": state.local_window.tolist(),  # 5×5×13

            # Global context
            "cycle": state.current_cycle,
            "stv": state.current_stv,
            "active_area_density": state.active_area_density,
            "route_congestion": state.route_congestion,
            "factory_load": state.factory_load,
            "global_distance_ratio": state.global_distance_ratio,

            # Candidate neighbors
            "neighbors": [list(n) for n in state.neighbors],
            "neighbor_costs": state.neighbor_costs,

            # Chosen action
            "chosen": list(chosen_neighbor),
            "chosen_cost": chosen_cost
        }

        self.file.write(json.dumps(record) + '\n')
        self.file.flush()
        self.expansion_count += 1

    def log_search_complete(self, final_cost: float, path_length: int,
                           found: bool):
        """Log search outcome (for reconstruction)"""
        if not self.enabled:
            return

        record = {
            "type": "complete",
            "search_id": self.search_id,
            "final_cost": final_cost,
            "path_length": path_length,
            "found": found,
            "n_expansions": self.expansion_count
        }

        self.file.write(json.dumps(record) + '\n')
        self.file.flush()

        # Reset for next search
        self.search_id += 1
        self.expansion_count = 0
