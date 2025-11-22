#!/usr/bin/env python3
"""
Example script for running ML-guided A* routing

This demonstrates:
1. Baseline compilation (use_ml=False)
2. Data generation with JSONL logging (log_jsonl=True)
3. ML-guided compilation (use_ml=True, with trained model)
"""

from surface_code_routing.dag import DAG
from surface_code_routing.instructions import Hadamard, CNOT, INIT
from surface_code_routing.compiled_qcb import compile_qcb


def test_baseline_vs_ml():
    """Compare baseline A* vs ML-guided A*"""

    # Build simple GHZ circuit
    dag = DAG('GHZ4')
    dag.add_gate(Hadamard('q_0'))
    for i in range(1, 4):
        dag.add_gate(CNOT('q_0', f'q_{i}'))

    # Baseline compilation
    print("=" * 60)
    print("Running baseline A* (no ML)...")
    print("=" * 60)
    baseline = compile_qcb(
        dag, 10, 10,
        use_ml=False,
        verbose=True
    )

    print(f"\nBaseline Results:")
    print(f"  Cycles: {baseline.n_cycles()}")
    print(f"  STV: {baseline.space_time_volume()}")

    # ML-guided compilation (requires trained model)
    # Uncomment when you have a trained model
    """
    print("\n" + "=" * 60)
    print("Running ML-guided A*...")
    print("=" * 60)
    ml_guided = compile_qcb(
        dag, 10, 10,
        use_ml=True,
        nn_model_path="./models/routing_gnn.pt",
        verbose=True
    )

    print(f"\nML-Guided Results:")
    print(f"  Cycles: {ml_guided.n_cycles()}")
    print(f"  STV: {ml_guided.space_time_volume()}")

    # Comparison
    stv_improvement = (baseline.space_time_volume() - ml_guided.space_time_volume()) / baseline.space_time_volume() * 100
    print(f"\nImprovement: {stv_improvement:.2f}% STV reduction")
    """


def generate_training_data():
    """Generate JSONL training data for various circuits"""

    print("=" * 60)
    print("Generating training data...")
    print("=" * 60)

    # GHZ circuits
    for n_qubits in range(3, 8):
        print(f"\nGenerating GHZ({n_qubits})...")
        dag = DAG(f'GHZ{n_qubits}')
        dag.add_gate(Hadamard('q_0'))
        for i in range(1, n_qubits):
            dag.add_gate(CNOT('q_0', f'q_{i}'))

        result = compile_qcb(
            dag, 16, 16,
            log_jsonl=True,
            jsonl_path=f"./data/ghz_{n_qubits}.jsonl",
            verbose=False
        )

        print(f"  Cycles: {result.n_cycles()}, STV: {result.space_time_volume()}")

    print("\nTraining data generation complete!")
    print("JSONL files saved in ./data/")


def test_simple_circuit_with_logging():
    """Test a simple circuit with JSONL logging enabled"""

    print("=" * 60)
    print("Testing with JSONL logging...")
    print("=" * 60)

    dag = DAG('Simple')
    dag.add_gate(INIT('q_0', 'q_1', 'q_2'))
    dag.add_gate(Hadamard('q_0'))
    dag.add_gate(CNOT('q_0', 'q_1'))
    dag.add_gate(CNOT('q_1', 'q_2'))

    result = compile_qcb(
        dag, 8, 8,
        log_jsonl=True,
        jsonl_path="./test_routing.jsonl",
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Cycles: {result.n_cycles()}")
    print(f"  STV: {result.space_time_volume()}")
    print(f"\nRouting log saved to: test_routing.jsonl")


if __name__ == "__main__":
    import sys
    import os

    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "baseline":
            test_baseline_vs_ml()
        elif mode == "generate":
            generate_training_data()
        elif mode == "log":
            test_simple_circuit_with_logging()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python run_ml_guided.py [baseline|generate|log]")
    else:
        # Default: run baseline test
        print("Running baseline test (use 'generate' to create training data)")
        print()
        test_baseline_vs_ml()
