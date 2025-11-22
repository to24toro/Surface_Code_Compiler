import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MultiHeadRoutingGNN(nn.Module):
    """
    Multi-head GNN for routing guidance

    Architecture:
    - GNN encoder (3-layer GCN)
    - Global feature encoder
    - Three heads:
        * value: scalar cost-to-go estimate
        * prune: per-neighbor pruning score [0,1]
        * policy: per-neighbor policy logit (raw)
    """

    def __init__(self, node_feat_dim=13, global_feat_dim=12, hidden_dim=128):
        super().__init__()

        # GNN encoder
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Global feature encoder
        self.fc_global = nn.Linear(global_feat_dim, hidden_dim)

        # Three heads
        self.value_head = nn.Linear(hidden_dim * 2, 1)  # Combined embedding → scalar
        self.prune_head = nn.Linear(hidden_dim, 1)      # Per-node → prune score
        self.policy_head = nn.Linear(hidden_dim, 1)     # Per-node → policy logit

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        u = data.u
        neighbor_mask = data.neighbor_mask
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode graph structure
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        node_embeddings = F.relu(self.conv3(x, edge_index))

        # Global graph embedding
        graph_embedding = global_mean_pool(node_embeddings, batch)

        # Global features
        global_feat = F.relu(self.fc_global(u))

        # Combined embedding for value head
        combined = torch.cat([graph_embedding, global_feat], dim=-1)

        # Value head (scalar cost-to-go)
        value = self.value_head(combined)

        # Prune and policy heads (per-neighbor scores)
        # Extract only neighbor nodes
        neighbor_embeddings = node_embeddings[neighbor_mask]

        if neighbor_embeddings.size(0) > 0:
            # Prune scores: sigmoid to [0,1]
            prune_scores = torch.sigmoid(self.prune_head(neighbor_embeddings).squeeze(-1))
            # Policy logits: raw scores (no softmax)
            policy_logits = self.policy_head(neighbor_embeddings).squeeze(-1)
        else:
            # No neighbors (shouldn't happen in practice)
            prune_scores = torch.zeros(0, device=x.device)
            policy_logits = torch.zeros(0, device=x.device)

        return {
            'value': value.squeeze(-1),
            'prune': prune_scores,
            'policy': policy_logits
        }
