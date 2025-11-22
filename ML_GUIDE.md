# ML-Guided A* for Lattice Surgery Routing

This implementation adds machine learning guidance to the A* search in the Surface Code Compiler.

## Overview

The ML-guided A* system provides three types of guidance:
- **Value**: Cost-to-go estimation (replaces heuristic)
- **Pruning**: Early detection of bad branches
- **Policy**: Ranking of candidate actions

**Important**: ML is a meta-heuristic only. All correctness and fault-tolerance logic remains classical.

## Installation

The base package is already installed. For ML features, you need:

```bash
pip install torch torch-geometric
```

## Quick Start

### 1. Baseline Compilation (No ML)

```python
from surface_code_routing.dag import DAG
from surface_code_routing.instructions import Hadamard, CNOT
from surface_code_routing.compiled_qcb import compile_qcb

dag = DAG('GHZ4')
dag.add_gate(Hadamard('q_0'))
for i in range(1, 4):
    dag.add_gate(CNOT('q_0', f'q_{i}'))

result = compile_qcb(dag, 10, 10, use_ml=False)
print(f"Cycles: {result.n_cycles()}, STV: {result.space_time_volume()}")
```

### 2. Generate Training Data

```python
# Generate JSONL logs during compilation
result = compile_qcb(
    dag, 10, 10,
    log_jsonl=True,
    jsonl_path="./data/ghz_4.jsonl"
)
```

Each A* expansion is logged with:
- Current state (5×5 local window, 13D features per patch)
- Global context (cycle, STV, density, congestion, etc.)
- Candidate neighbors
- Chosen action

### 3. Train the Model

```python
# Train GNN model (example)
from surface_code_routing.ml.model import MultiHeadRoutingGNN
import torch
from torch_geometric.data import DataLoader

# Load JSONL data and create PyG dataset
# ... (training code to be implemented)

model = MultiHeadRoutingGNN(node_feat_dim=13, global_feat_dim=12, hidden_dim=128)
# ... train model ...
torch.save(model, "./models/routing_gnn.pt")
```

### 4. Use ML-Guided A*

```python
result = compile_qcb(
    dag, 10, 10,
    use_ml=True,
    nn_model_path="./models/routing_gnn.pt"
)
```

## Architecture

### State Representation (Minimal)

`RoutingState` contains:
- **Local window**: 5×5 grid around current node (13D features per patch)
- **Global features**:
  - Active area density
  - Route congestion (7×7 region)
  - Factory load
  - Global distance ratio
  - Cycle, STV, active/waiting gates
- **Candidates**: List of neighbor nodes to expand

**Size**: ~1-2 KB per state (NOT full grid)

### Patch Features (13D)

Each patch in the 5×5 window:
- [0-4]: Patch type one-hot (ROUTE, REG, IO, EXTERN, NONE)
- [5-6]: Orientation one-hot (X, Z)
- [7]: Is locked
- [8]: Locked by current gate
- [9]: Locked by other gate
- [10]: Active flag
- [11]: Distance to goal (normalized)
- [12]: Reserved

### Graph Serialization

Converts `RoutingState` → JSON with:
- **Nodes**: ≤25 nodes (5×5 local window)
- **Edges**: 4-connectivity
- **Global**: 12D feature vector

### GNN Model

`MultiHeadRoutingGNN`:
- **Encoder**: 3-layer GCN
- **Heads**:
  - `value_head`: Scalar cost-to-go
  - `prune_head`: Per-neighbor pruning score [0,1]
  - `policy_head`: Per-neighbor policy logit (raw)

### Pruning Logic

Enhanced pruning with goal-directional boost:
```python
effective_prune_score = nn_prune_score
if neighbor_dist_to_goal > current_dist_to_goal:
    effective_prune_score += 0.2

if effective_prune_score > 0.5:
    prune()
```

## File Structure

```
Surface_Code_Compiler/
├── src/surface_code_routing/
│   ├── routing_state.py          # Minimal state representation
│   ├── graph_serialization.py    # State → graph JSON
│   ├── jsonl_logger.py           # Training data logger
│   ├── nn_client.py              # ML model interface
│   ├── stv_utils.py              # STV utilities
│   ├── circuit_model.py          # Modified A* (route method)
│   ├── router.py                 # Modified router
│   ├── compiled_qcb.py           # Modified compile_qcb
│   └── ml/
│       └── model.py              # GNN model definition
├── examples/
│   └── run_ml_guided.py          # Example usage
└── data/                         # Training data (JSONL)
```

## Example Scripts

### Baseline Test
```bash
python3.12 examples/run_ml_guided.py baseline
```

### Generate Training Data
```bash
python3.12 examples/run_ml_guided.py generate
```

### Test with Logging
```bash
python3.12 examples/run_ml_guided.py log
```

## Constraints & Design Decisions

### Why Minimal State?

**Problem**: Full grid serialization would be ~100 KB per expansion
**Solution**: 5×5 local window (~2 KB per expansion)

### Why One Forward Pass?

**Problem**: Calling NN for each candidate would be too slow
**Solution**: Single forward pass, multi-head output for all candidates

### Why Raw Policy Logits?

**Problem**: Softmax assumes mutual exclusivity
**Solution**: Raw logits allow independent scoring

### Why 0.5 Prune Threshold?

**Problem**: Fixed 0.9 threshold too conservative
**Solution**: 0.5 with goal-directional boost (+0.2 if moving away)

## Training Data Format

JSONL log structure:
```json
{
  "type": "expansion",
  "search_id": 0,
  "expansion_id": 5,
  "current": [2, 3],
  "goal": [8, 7],
  "start": [0, 1],
  "g_cost": 5.0,
  "h_cost": 10.0,
  "local_window": [[[...], ...], ...],  // 5×5×13
  "cycle": 3,
  "stv": 150,
  "active_area_density": 0.45,
  "route_congestion": 12.0,
  "factory_load": 2,
  "global_distance_ratio": 0.15,
  "neighbors": [[2, 4], [3, 3]],
  "neighbor_costs": [6.0, 6.0],
  "chosen": [2, 4],
  "chosen_cost": 6.0
}
```

## Backward Compatibility

When `use_ml=False` (default):
- No ML client instantiation
- No JSONL logging overhead
- Identical behavior to original A*
- Same results, same performance

## Performance Expectations

**Without ML**:
- Standard A* with Manhattan heuristic
- Proven correctness

**With ML** (after training):
- Reduced STV (better routing)
- Fewer cycles
- Fewer explored nodes
- Faster search (with pruning)
- Same correctness guarantees

## Next Steps

1. **Generate data**: Run baseline on various circuits
2. **Train model**: Implement training loop for GNN
3. **Evaluate**: Compare ML-guided vs baseline on benchmarks
4. **Tune**: Adjust prune threshold, policy weighting, etc.

## Research Goals

Demonstrate that ML-guided A* achieves:
- Lower space-time volume
- Fewer cycles
- Fewer explored nodes
- Faster scheduling time

While maintaining:
- Full correctness
- Fault-tolerance guarantees
- Classical verification

## Citation

Based on the Surface Code Compiler:
https://github.com/Alan-Robertson/Surface_Code_Compiler
