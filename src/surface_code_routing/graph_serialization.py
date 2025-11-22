def state_to_graph_json(state: 'RoutingState') -> dict:
    """
    Convert MINIMAL state to graph JSON

    Graph contains:
    - Nodes: 5×5 local window (≤25 nodes)
    - Edges: grid connectivity (4-connectivity)
    - Global features: enhanced scalars
    """
    nodes = []
    edges = []

    # Build local 5×5 window centered on current
    cy, cx = state.current_pos
    gy, gx = state.goal_pos
    sy, sx = state.start_pos

    for dy in range(-2, 3):  # -2, -1, 0, 1, 2
        for dx in range(-2, 3):
            y, x = cy + dy, cx + dx

            # Skip out-of-bounds
            if not (0 <= y < state.grid_shape[0] and 0 <= x < state.grid_shape[1]):
                continue

            # Get features from local_window
            feat = state.local_window[dy + 2, dx + 2]  # Shift to [0,4]

            node = {
                "id": f"p_{y}_{x}",
                "coords": [y, x],
                "features": feat.tolist(),  # 13 floats
                "is_current": (y == cy and x == cx),
                "is_goal": (y == gy and x == gx),
                "is_start": (y == sy and x == sx),
            }
            nodes.append(node)

            # Add edges to neighbors (4-connectivity)
            # Right edge
            if dx < 2 and (0 <= y < state.grid_shape[0] and 0 <= x + 1 < state.grid_shape[1]):
                edges.append({
                    "source": f"p_{y}_{x}",
                    "target": f"p_{y}_{x+1}",
                    "type": "horizontal"
                })
            # Down edge
            if dy < 2 and (0 <= y + 1 < state.grid_shape[0] and 0 <= x < state.grid_shape[1]):
                edges.append({
                    "source": f"p_{y}_{x}",
                    "target": f"p_{y+1}_{x}",
                    "type": "vertical"
                })

    return {
        "nodes": nodes,  # ≤25 nodes
        "edges": edges,  # ≤48 edges
        "global": {
            "grid_shape": list(state.grid_shape),
            "current_cycle": state.current_cycle,
            "current_stv": state.current_stv,
            "n_active_gates": state.n_active_gates,
            "n_waiting_gates": state.n_waiting_gates,
            "g_cost": state.g_cost,
            "h_cost": state.h_cost,
            "manhattan_dist": abs(cy - gy) + abs(cx - gx),
            # Enhanced features
            "active_area_density": state.active_area_density,
            "route_congestion": state.route_congestion,
            "factory_load": state.factory_load,
            "global_distance_ratio": state.global_distance_ratio,
        }
    }
