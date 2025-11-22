from typing import Optional, Dict, Any, List, Tuple


class NnClient:
    """
    Client for ML-guided A* heuristics

    Provides:
    - value: cost-to-go estimate
    - prune_scores: per-neighbor pruning scores [0,1]
    - policy_logits: per-neighbor policy scores (raw logits)
    """

    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cpu', enabled: bool = False):
        self.enabled = enabled
        self.model = None
        self.device = device

        if enabled and model_path:
            import torch
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def forward(self, state: 'RoutingState') -> Dict[str, Any]:
        """
        ONE forward pass through GNN

        Returns:
        {
            'value': float,  # Cost-to-go estimate
            'prune_scores': List[float],  # One per neighbor [0,1]
            'policy_logits': List[float]  # One per neighbor (raw logits)
        }
        """
        if not self.enabled or self.model is None:
            # Fallback: no ML guidance
            n = len(state.neighbors)
            return {
                'value': 0.0,
                'prune_scores': [0.0] * n,
                'policy_logits': [0.0] * n  # Equal priority
            }

        # Import here to avoid dependency when ML disabled
        from surface_code_routing.graph_serialization import state_to_graph_json

        # Convert state to PyG Data
        graph_json = state_to_graph_json(state)
        data = self._json_to_pyg(graph_json, state.neighbors, state.current_pos)

        import torch
        with torch.no_grad():
            output = self.model(data)

        return {
            'value': float(output['value'].item()),
            'prune_scores': output['prune'].cpu().numpy().tolist(),
            'policy_logits': output['policy'].cpu().numpy().tolist()
        }

    def _json_to_pyg(self, graph_json: dict, neighbors: List[Tuple[int, int]],
                     current_pos: Tuple[int, int]):
        """Convert minimal graph to PyG Data"""
        import torch
        from torch_geometric.data import Data

        # Node features (13D)
        node_features = [node['features'] for node in graph_json['nodes']]
        x = torch.tensor(node_features, dtype=torch.float32)

        # Edge index
        node_id_to_idx = {n['id']: i for i, n in enumerate(graph_json['nodes'])}
        edge_list = []
        for edge in graph_json['edges']:
            if edge['source'] in node_id_to_idx and edge['target'] in node_id_to_idx:
                src = node_id_to_idx[edge['source']]
                tgt = node_id_to_idx[edge['target']]
                edge_list.append([src, tgt])
                edge_list.append([tgt, src])  # Undirected

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Global features (12D)
        g = graph_json['global']
        global_feat = [
            g['grid_shape'][0] / 100.0,
            g['grid_shape'][1] / 100.0,
            g['current_cycle'] / 1000.0,
            g['current_stv'] / 10000.0,
            g['n_active_gates'] / 10.0,
            g['n_waiting_gates'] / 10.0,
            g['g_cost'] / 100.0,
            g['manhattan_dist'] / 100.0,
            g['active_area_density'],
            g['route_congestion'] / 49.0,  # Normalize by 7*7
            g['factory_load'] / 10.0,
            g['global_distance_ratio']
        ]
        u = torch.tensor(global_feat, dtype=torch.float32).unsqueeze(0)

        # Neighbor mask (which nodes are candidates)
        neighbor_mask = torch.zeros(len(node_features), dtype=torch.bool)
        neighbor_indices = []
        for (ny, nx) in neighbors:
            node_id = f"p_{ny}_{nx}"
            if node_id in node_id_to_idx:
                idx = node_id_to_idx[node_id]
                neighbor_mask[idx] = True
                neighbor_indices.append(idx)

        # Current node index
        current_id = f"p_{current_pos[0]}_{current_pos[1]}"
        current_idx = node_id_to_idx.get(current_id, 0)

        return Data(
            x=x,
            edge_index=edge_index,
            u=u,
            neighbor_mask=neighbor_mask,
            neighbor_indices=torch.tensor(neighbor_indices, dtype=torch.long),
            current_idx=torch.tensor([current_idx], dtype=torch.long)
        )
