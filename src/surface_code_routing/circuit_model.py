"""
    circuit_model
    Provides a graph as a regular lattice of surface code patches
"""

import queue
import numpy as np

from surface_code_routing.qcb import SCPatch
from surface_code_routing.tikz_utils import tikz_patch_graph
from surface_code_routing.utils import debug_print
from surface_code_routing.bind import AddrBind

from surface_code_routing.constants import SINGLE_ANCILLAE, ELBOW_ANCILLAE


class PatchGraphNode:
    """
    Single surface code patch
    """

    INITIAL_LOCK_STATE = AddrBind("INITIAL LOCK STATE")
    Z_ORIENTED = AddrBind("Z")  # Smooth edge up
    X_ORIENTED = AddrBind("X")  # Rough edge up
    SUGGEST_ROUTE = AddrBind("Suggest Route")
    SUGGEST_ROTATE = AddrBind("Suggest Rotate")
    ANCILLAE_STATES = {SCPatch.ROUTE, SCPatch.LOCAL_ROUTE}
    SCOPE_STATES = {SCPatch.REG, SCPatch.IO}


    def __init__(
        self, graph, i: int, j: int, orientation: AddrBind = None, verbose: bool = False
    ):
        self.graph = graph
        self.y = i
        self.x = j
        self.state = SCPatch.ROUTE
        self.last_used = -1

        if orientation is None:
            orientation = self.X_ORIENTED
        self.orientation = orientation
        self.lock_state = self.INITIAL_LOCK_STATE

        self.verbose = verbose

    def set_underlying(self, state):
        """
        Sets the underlying state of the patch
        """
        debug_print(self, state, debug=self.verbose)
        self.state = state

    def adjacent(self, gate, **kwargs) -> set:
        """
        Returns adjacent elements
        """
        return self.graph.adjacent(self.y, self.x, gate, **kwargs)

    def anc_check(self, anc, gate, unique: bool = True):
        """
        Checks if this patch qualifies for use as an ancillae
        """
        if anc.state not in PatchGraphNode.ANCILLAE_STATES or not anc.probe(gate, unique=unique):
            return None
        return anc

    def anc_above(self, gate, unique: bool = True):
        """
        Checks the patch above for ancillae use
        """
        if self.y == 0:
            return None
        anc = self.graph[self.y - 1, self.x]
        return self.anc_check(anc, gate, unique=unique)

    def anc_below(self, gate, unique: bool = True):
        """
        Checks the patch below for ancillae use
        """
        if self.y == self.graph.graph.shape[0] - 1:
            return None
        anc = self.graph[self.y + 1, self.x]
        return self.anc_check(anc, gate, unique=unique)

    def anc_left(self, gate, unique=True):
        """
        Checks the patch to the left for ancillae use
        """
        if self.x == 0:
            return None
        anc = self.graph[self.y, self.x - 1]
        return self.anc_check(anc, gate, unique=unique)

    def anc_right(self, gate, unique=True):
        """
        Checks the patch to the right for ancillae use
        """
        if self.x == self.graph.graph.shape[1] - 1:
            return None
        anc = self.graph[self.y, self.x + 1]
        return self.anc_check(anc, gate, unique=unique)

    def anc_vertical(self, gate):
        """
        Checks above and below for ancillae use
        """
        ancs = self.anc_above(gate), self.anc_below(gate)
        return filter(lambda i: i is not None, ancs)

    def anc_horizontal(self, gate):
        """
        Checks left and right for ancillae use
        """

        ancs = self.anc_left(gate), self.anc_right(gate)
        return filter(lambda i: i is not None, ancs)

    def __gt__(self, *args) -> bool:
        return 1

    def __repr__(self) -> str:
        return str(f"[{self.y}, {self.x}]")

    def __str__(self) -> str:
        return self.__repr__()

    def cost(self):
        """
        Currently unused
        """
        return 1

    def active_gates(self) -> set:
        """
        Wrapper around the graph's active gates
        """
        return self.graph.active_gates()

    def valid_edge(self, other_patch, edge):
        """
        Wrapper around the qcb valid edge call
        """
        return self.state.valid_edge(other_patch.state, edge)

    def probe(self, lock_request, unique=False) -> bool:
        """
        Probes whether this patch is locked or may be locked by the given gate
        """
        if not unique and self.lock_state is lock_request:
            return True
        if unique and self.lock_state is lock_request:
            return False
        # Gate has completed and is no longer active
        if self.lock_state not in self.active_gates():
            return True
        return False

    def lock(self, dag_node):
        """
        Locks the patch for use by a dag node
        """
        if probe := self.probe(dag_node):
            self.lock_state = dag_node
        return probe

    def route_or_rotate(self, orientation) -> AddrBind:
        """
        Determines if this patch should be used for routing or rotating
        """
        if orientation == self.orientation:
            debug_print("MATCHING ORIETATION", self, debug=self.verbose)
            return self.SUGGEST_ROUTE
        if (
            next(self.adjacent(None, bound=False, vertical=False, probe=False), None)
            is not None
        ):
            debug_print(
                "HORIZONTAL_ROUTING",
                self,
                tuple(self.adjacent(None, bound=False, vertical=False, probe=False)),
                debug=self.verbose,
            )

            return self.SUGGEST_ROUTE
        debug_print("FALLBACK ROTATE", self, debug=self.verbose)
        return self.SUGGEST_ROTATE

    def rotate(self):
        """
        Flip patch orientation
        """
        if self.orientation == self.Z_ORIENTED:
            self.orientation = self.X_ORIENTED
        else:
            self.orientation = self.Z_ORIENTED


class PatchGraph:
    """
    Graph of patches
    """

    VOLUME_PROBE = object()
    NO_PATH_FOUND = object()

    def __init__(
        self,
        shape: tuple,
        mapper,
        environment,
        default_orientation=PatchGraphNode.X_ORIENTED,
        verbose: bool = False,
    ):
        self.shape = shape
        self.environment = environment
        self.mapper = mapper
        self.default_orientation = default_orientation

        self.verbose = verbose

        self.graph = np.array(
            [
                [
                    PatchGraphNode(
                        self,
                        i,
                        j,
                        orientation=self.default_orientation,
                        verbose=self.verbose,
                    )
                    for j in range(shape[1])
                ]
                for i in range(shape[0])
            ]
        )

        for segment in self.mapper.map.values():
            for coordinates in segment.range():
                if self.graph[coordinates].state == SCPatch.ROUTE:
                    self.graph[coordinates].set_underlying(segment.get_slot())

        local_patches = []
        for segment in self.mapper.qcb:
            if segment.get_state() == SCPatch.LOCAL_ROUTE:
                for coordinates in segment.range():
                    self.graph[coordinates].set_underlying(SCPatch.LOCAL_ROUTE)
                    local_patches.append(self.graph[coordinates])

        # See if we can't eliminate some local patches
        # TODO BFS this
        expand = True
        while expand:
            expand = False
            uncleared_patches = []
            for local_patch in local_patches:
                if any(
                    p.state is SCPatch.ROUTE
                    for p in local_patch.adjacent(None, bound=False, probe=False)
                ):
                    expand = True
                    local_patch.set_underlying(SCPatch.ROUTE)
                else:
                    uncleared_patches.append(local_patch)
            local_patches = uncleared_patches

    def space_time_volume(self) -> int: 
        '''
            Counts the number of in-use patches
        '''
        volume = 0
        for row in self.graph: 
            for patch in row:
                if patch.state in PatchGraphNode.ANCILLAE_STATES:
                    if not patch.probe(PatchGraph.VOLUME_PROBE):
                        volume += 1
                elif patch.state in PatchGraphNode.SCOPE_STATES:
                    volume += 1
        return volume


    def debug_print(self, *args, **kwargs):
        """
        Debug print wrapper
        """
        debug_print(*args, **kwargs, debug=self.verbose)

    def __getitem__(self, coords) -> PatchGraphNode:
        return self.graph.__getitem__(tuple(coords))

    def active_gates(self) -> set:
        """
        Returns the set of active gates
        """
        return self.environment.active_gates

    def adjacent(
        self,
        i,
        j,
        gate,
        bound=True,  # Not constrained by initial graph node state
        horizontal=True,  # Checks horizonal
        vertical=True,  # Checks vertical
        probe=True,  # Probes for locking
        orientation=None,  # Constrains on orientation
    ):
        """
        Returns adjacent patches subject to constraints
        """
        opt = []

        if orientation is not None:
            if orientation == self.graph[i, j].orientation:
                horizontal = False
                bound = False
            else:
                vertical = False
                bound = False

        if horizontal and (not bound or self.graph[i, j].state is SCPatch.ROUTE):
            if j + 1 < self.shape[1]:
                if self.graph[i, j + 1].state is SCPatch.ROUTE:
                    opt.append([i, j + 1])

            if j - 1 >= 0:
                if self.graph[i, j - 1].state is SCPatch.ROUTE:
                    opt.append([i, j - 1])

        if vertical:
            if i + 1 < self.shape[0]:
                if (self.graph[i, j].state is SCPatch.ROUTE) or (
                    self.graph[i + 1, j].state is SCPatch.ROUTE
                ):
                    opt.append([i + 1, j])

            if i - 1 >= 0:
                if (self.graph[i, j].state is SCPatch.ROUTE) or (
                    self.graph[i - 1, j].state is SCPatch.ROUTE
                ):
                    opt.append([i - 1, j])
        for idx in opt:
            # Return without worrying about locks
            if probe is False:
                yield self[tuple(idx)]
            elif self[tuple(idx)].probe(gate):
                yield self[tuple(idx)]

    def flush(self):
        """
            Force unlock all routes
        """
        for grp in self.graph:
            for node in grp:
                if node.state in PatchGraphNode.ANCILLAE_STATES:
                    node.lock_state = PatchGraphNode.INITIAL_LOCK_STATE

    def route(
        self,
        start,  # Start node
        end,  # End node
        gate,  # Gate object
        heuristic=None,  # A* cost heuristic
        track_rotations=True,  # Rotation tracking
        start_orientation=None,  # Orietnation of start node
        end_orientation=None,  # Orientation of end node
        nn_client=None,  # ML client
        jsonl_logger=None,  # JSONL logger
        router_context=None  # Router context for global features
    ):
        """
        ML-augmented A* (MINIMAL state)

        Improvements:
        - Enhanced pruning with goal-directional boost
        - Policy uses raw logits (no softmax)
        - Value head replaces heuristic when ML enabled
        """

        if heuristic is None:
            heuristic = self.heuristic

        frontier = queue.PriorityQueue()
        frontier.put((0, start))

        path = {}
        path_cost = {}
        path[start] = None
        path_cost[start] = 0

        orientation = None

        while not frontier.empty():
            current = frontier.get()[1]

            if current == end:
                break

            # Orientation handling
            if track_rotations and current == start:
                orientation = start_orientation
                debug_print(
                    current,
                    gate,
                    orientation,
                    current.adjacent(gate, orientation=orientation),
                    debug=self.verbose,
                )

            # Get adjacent nodes (candidates)
            adj_nodes = list(current.adjacent(gate, orientation=orientation))

            if len(adj_nodes) == 0:
                if current == start:
                    orientation = None
                continue

            # Build MINIMAL state (ONCE per expansion)
            if nn_client and nn_client.enabled:
                routing_state = self._build_minimal_state(
                    current, start, end, path_cost,
                    adj_nodes, router_context, gate
                )

                # ONE forward pass
                nn_output = nn_client.forward(routing_state)
                value_estimate = nn_output['value']
                prune_scores = nn_output['prune_scores']
                policy_logits = nn_output['policy_logits']
            else:
                value_estimate = 0.0
                prune_scores = [0.0] * len(adj_nodes)
                policy_logits = [0.0] * len(adj_nodes)

            # Sort neighbors by policy logits (descending, raw logits)
            sorted_neighbors = sorted(
                zip(adj_nodes, prune_scores, policy_logits),
                key=lambda x: -x[2]  # Descending by policy logit
            )

            # Track if we expanded any neighbor
            expanded_any = False
            chosen_neighbor = None
            chosen_cost = None

            # Expand neighbors
            current_dist_to_goal = abs(current.y - end.y) + abs(current.x - end.x)

            for neighbor, prune_score, policy_logit in sorted_neighbors:
                # Enhanced pruning logic
                # Base pruning score from NN
                effective_prune_score = prune_score

                # Boost prune score if neighbor increases Manhattan distance to goal
                neighbor_dist_to_goal = abs(neighbor.y - end.y) + abs(neighbor.x - end.x)
                if neighbor_dist_to_goal > current_dist_to_goal:
                    effective_prune_score += 0.2

                # Prune if effective score > 0.5
                if effective_prune_score > 0.5:
                    continue

                if (neighbor == end and current != start) or neighbor.state == SCPatch.ROUTE:
                    cost = path_cost[current] + neighbor.cost()

                    # Use ML value estimate OR classical heuristic
                    if nn_client and nn_client.enabled:
                        h_value = value_estimate
                    else:
                        h_value = heuristic(neighbor, end)

                    if neighbor not in path_cost or cost < path_cost[neighbor]:
                        path_cost[neighbor] = cost
                        frontier.put((cost + h_value, neighbor))
                        path[neighbor] = current

                        # Record first successful expansion for logging
                        if not expanded_any:
                            expanded_any = True
                            chosen_neighbor = (neighbor.y, neighbor.x)
                            chosen_cost = cost

            # Log expansion (ONCE, with chosen neighbor)
            if jsonl_logger and jsonl_logger.enabled and nn_client and nn_client.enabled:
                if chosen_neighbor is not None:
                    jsonl_logger.log_expansion(
                        state=routing_state,
                        chosen_neighbor=chosen_neighbor,
                        chosen_cost=chosen_cost
                    )

            if current == start:
                orientation = None

        else:
            # No path found
            if jsonl_logger and jsonl_logger.enabled:
                jsonl_logger.log_search_complete(
                    final_cost=float('inf'),
                    path_length=0,
                    found=False
                )
            return self.NO_PATH_FOUND

        # Path found
        final_route = self._traverse_path(path, end)[::-1] + [end]

        if jsonl_logger and jsonl_logger.enabled:
            jsonl_logger.log_search_complete(
                final_cost=path_cost[end],
                path_length=len(final_route),
                found=True
            )

        return final_route

    def ancillae(self, gate, start, n_ancillae):
        """
        ancillae
        Attempts to lock ancillae for a gate
        """
        # Currently supports a single ancillae
        if gate.ancillae_type() == SINGLE_ANCILLAE:
            return self.single_ancillae(gate, start)
        if gate.ancillae_type() == ELBOW_ANCILLAE:
            return self.elbow_ancillae(gate, start)
        # This is never reached
        return None

    def single_ancillae(self, gate, start):
        """
        Locks a single ancillae
        """
        potential_ancillae = start.adjacent(gate, bound=False)
        for anc in potential_ancillae:
            if anc.probe(gate) and anc.lock_state is not gate:
                return [anc]
        return self.NO_PATH_FOUND

    def ancillae_elbow_path(self, start, gate, gen_function, transverse_function):
        """
        Locks an elbow ancillae
        """
        ancillae = []
        curr = start
        while curr is not None:
            ancillae.append(curr)
            transverse_ancillae = transverse_function(curr, gate)
            transverse_ancilla = next(iter(transverse_ancillae), None)
            if transverse_ancilla is not None:
                ancillae.append(transverse_ancilla)
                return ancillae
            curr = gen_function(curr, gate)
        return None

    def elbow_ancillae(self, gate, start):
        """
        ## #   # ##
        #  ## ##  #
        """
        # Try local deformation first
        h_anc = next(iter(anc for anc in start.anc_horizontal(gate)), None)
        v_anc = next(iter(anc for anc in start.anc_vertical(gate)), None)

        if h_anc is not None and v_anc is not None:
            return [h_anc, start, v_anc]

        for generative_fn, transverse_fn in zip(
            [
                PatchGraphNode.anc_above,
                PatchGraphNode.anc_below,
                PatchGraphNode.anc_left,
                PatchGraphNode.anc_right,
            ],
            [
                PatchGraphNode.anc_horizontal,
                PatchGraphNode.anc_horizontal,
                PatchGraphNode.anc_vertical,
                PatchGraphNode.anc_vertical,
            ],
        ):
            ancillae = self.ancillae_elbow_path(
                generative_fn(start, gate), gate, generative_fn, transverse_fn
            )
            if ancillae is not None:
                return ancillae

        return self.NO_PATH_FOUND

    def l_ancillae(self, gate, start):
        """
        Gets potential unlocked ancillae
        """
        potential_ancillae = start.adjacent(gate, bound=False)
        for anc in potential_ancillae:
            if anc.lock_state is not gate:
                return [anc]
        return self.NO_PATH_FOUND

    def _encode_patch_features(self, patch, goal_pos, current_gate) -> np.ndarray:
        """
        Encode single patch to 13D feature vector

        Features:
        [0-4]: patch type (one-hot: ROUTE, REG, IO, EXTERN, NONE)
        [5-6]: orientation (one-hot: X, Z)
        [7]: is_locked (any lock)
        [8]: locked_by_self (locked by current gate)
        [9]: locked_by_other (locked by different gate)
        [10]: active_flag (in ancilla/scope states)
        [11]: distance_to_goal (normalized)
        [12]: reserved
        """
        features = np.zeros(13, dtype=np.float32)

        # Patch type (one-hot, 5 dims)
        patch_types = [SCPatch.ROUTE, SCPatch.REG, SCPatch.IO,
                       SCPatch.EXTERN, SCPatch.NONE]
        for i, pt in enumerate(patch_types):
            if patch.state == pt:
                features[i] = 1.0

        # Orientation (one-hot, 2 dims)
        if patch.orientation == PatchGraphNode.X_ORIENTED:
            features[5] = 1.0
        else:
            features[6] = 1.0

        # Lock states
        is_locked = patch.lock_state != PatchGraphNode.INITIAL_LOCK_STATE
        features[7] = 1.0 if is_locked else 0.0

        # Locked by current gate
        if current_gate and patch.lock_state is current_gate:
            features[8] = 1.0

        # Locked by other gate
        if is_locked and current_gate and patch.lock_state is not current_gate:
            features[9] = 1.0

        # Active flag
        if patch.state in PatchGraphNode.ANCILLAE_STATES or patch.state in PatchGraphNode.SCOPE_STATES:
            features[10] = 1.0

        # Distance to goal (normalized)
        gy, gx = goal_pos
        dist = abs(patch.y - gy) + abs(patch.x - gx)
        features[11] = dist / 100.0  # Normalize

        # Reserved
        features[12] = 0.0

        return features

    def _build_minimal_state(self, current, start, end, path_cost,
                            adj_nodes, router_context, gate):
        """
        Build MINIMAL RoutingState (NO full grid dump)

        Computes enhanced global features:
        - active_area_density
        - route_congestion (7×7 region)
        - factory_load
        - global_distance_ratio
        """
        from surface_code_routing.routing_state import RoutingState

        cy, cx = current.y, current.x
        gy, gx = end.y, end.x
        sy, sx = start.y, start.x

        # Extract 5×5 local window
        local_window = np.zeros((5, 5, 13), dtype=np.float32)

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = cy + dy, cx + dx
                if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                    patch = self.graph[y, x]
                    features = self._encode_patch_features(patch, (gy, gx), gate)
                    local_window[dy + 2, dx + 2] = features

        # Get global context
        if router_context:
            current_cycle = len(router_context.layers)
            current_stv = router_context.space_time_volume
            n_active = len(router_context.active_gates)
            n_waiting = len([g for g in router_context.dag.gates
                            if g not in router_context.resolved])
        else:
            current_cycle = 0
            current_stv = 0
            n_active = 0
            n_waiting = 0

        # Compute enhanced global features
        # 1. Active area density
        total_patches = self.shape[0] * self.shape[1]
        active_patches = 0
        for row in self.graph:
            for patch in row:
                if patch.state in PatchGraphNode.ANCILLAE_STATES or patch.state in PatchGraphNode.SCOPE_STATES:
                    active_patches += 1
        active_area_density = active_patches / total_patches if total_patches > 0 else 0.0

        # 2. Route congestion (7×7 region centered at current)
        route_congestion = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y, x = cy + dy, cx + dx
                if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                    patch = self.graph[y, x]
                    if patch.lock_state != PatchGraphNode.INITIAL_LOCK_STATE:
                        route_congestion += 1

        # 3. Factory load
        if router_context:
            factory_load = len([g for g in router_context.active_gates if g.is_factory()])
        else:
            factory_load = 0

        # 4. Global distance ratio
        start_end_manhattan = abs(sy - gy) + abs(sx - gx)
        max_dimension = max(self.shape[0], self.shape[1])
        global_distance_ratio = start_end_manhattan / max_dimension if max_dimension > 0 else 0.0

        # Manhattan heuristic for h_cost (logging/training only)
        h_cost = abs(cy - gy) + abs(cx - gx)

        return RoutingState(
            current_pos=(cy, cx),
            goal_pos=(gy, gx),
            start_pos=(sy, sx),
            g_cost=path_cost.get(current, 0.0),
            h_cost=h_cost,
            local_window=local_window,
            grid_shape=self.shape,
            current_cycle=current_cycle,
            current_stv=current_stv,
            n_active_gates=n_active,
            n_waiting_gates=n_waiting,
            gate_id=str(gate.get_symbol()) if gate else "unknown",
            active_area_density=active_area_density,
            route_congestion=float(route_congestion),
            factory_load=factory_load,
            global_distance_ratio=global_distance_ratio,
            neighbors=[(n.y, n.x) for n in adj_nodes],
            neighbor_costs=[path_cost.get(current, 0.0) + n.cost() for n in adj_nodes]
        )

    def _traverse_path(self, path, end):
        """Helper to traverse path"""
        next_end = path[end]
        if next_end is not None:
            return [next_end] + self._traverse_path(path, next_end)
        return []

    @staticmethod
    def heuristic(a, b, bias=1 + 1e-7):
        """
        Default A* routing heuristic
        """
        return abs(a.x - b.x) + bias * abs(a.y - b.y)

    def __tikz__(self):
        return tikz_patch_graph(self)
