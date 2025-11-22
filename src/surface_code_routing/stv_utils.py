def compute_cycle_stv(graph: 'PatchGraph') -> int:
    """
    Count active patches for one cycle
    (Extracted from circuit_model.py:267-279)
    """
    from surface_code_routing.circuit_model import PatchGraphNode, PatchGraph

    volume = 0
    for row in graph.graph:
        for patch in row:
            if patch.state in PatchGraphNode.ANCILLAE_STATES:
                if not patch.probe(PatchGraph.VOLUME_PROBE):
                    volume += 1
            elif patch.state in PatchGraphNode.SCOPE_STATES:
                volume += 1
    return volume


def compute_total_stv(router: 'QCBRouter') -> int:
    """
    Total STV including externs
    (Extracted from compiled_qcb.py:173-185)
    """
    extern_volumes = sum(
        x.space_time_volume() for x in router.dag.externs.values()
    )
    return router.space_time_volume + extern_volumes
