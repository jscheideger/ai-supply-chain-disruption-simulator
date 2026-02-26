from __future__ import annotations

# dataclasses reduce boilerplate for stateful model objects.
# typing documents data shapes and helps catch mistakes early.
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# math supports an optional seasonal demand pattern.
# random supports synthetic variability and reproducible runs via seeding.
import math
import random


NodeId = str
EdgeId = Tuple[NodeId, NodeId]


@dataclass
class SupplyNode:
    """
    Represents a supply chain entity such as a supplier, plant, distribution center, or retail node.

    State tracked over time:
    inventory on hand, backlog (unmet demand), and policy thresholds.
    """
    node_id: NodeId
    node_type: str  # supplier, plant, dc, retail
    capacity: float
    holding_cost_per_unit: float
    shortage_cost_per_unit: float
    reorder_point: float
    order_up_to: float

    inventory_level: float = 0.0
    backlog: float = 0.0

    def updateInventory(self, qty: float) -> None:
        """
        Adjust inventory by qty.
        Positive adds inventory, negative removes inventory.

        Centralizing inventory updates helps keep state changes consistent and traceable.
        """
        self.inventory_level += qty

    def calculateStorageCost(self) -> float:
        """
        Compute holding cost for one time step based on current inventory.

        Holding cost supports cost based evaluation and highlights tradeoffs between inventory and service.
        """
        if self.inventory_level <= 0:
            return 0.0
        return self.inventory_level * self.holding_cost_per_unit

    def calculateShortageCost(self) -> float:
        """
        Compute shortage penalty cost for one time step based on current backlog.

        Shortage cost quantifies unmet demand impact and makes disruption effects measurable.
        """
        if self.backlog <= 0:
            return 0.0
        return self.backlog * self.shortage_cost_per_unit


@dataclass
class TransportationLane:
    """
    Represents a directed shipping lane between two nodes.

    Includes:
    capacity constraint per step, lead time, and per unit transportation cost.
    """
    sourceNode: NodeId
    destinationNode: NodeId
    capacity: float
    leadTime: int
    costPerUnit: float

    enabled: bool = True
    delaySteps: int = 0

    def transportGoods(self, qty: float, current_step: int) -> "Shipment":
        """
        Convert a shipped quantity into a Shipment with arrival time and transport cost.

        Lane lead time and temporary delays are applied here to keep timing logic in one place.
        """
        lead = int(self.leadTime + self.delaySteps)
        cost = qty * self.costPerUnit
        return Shipment(
            sourceNode=self.sourceNode,
            destinationNode=self.destinationNode,
            qty=qty,
            arrivalStep=current_step + lead,
            cost=cost,
        )


@dataclass
class Shipment:
    """
    Represents goods in transit from a source to a destination.

    Shipments are stored until their arrivalStep is reached, enforcing travel delays.
    """
    sourceNode: NodeId
    destinationNode: NodeId
    qty: float
    arrivalStep: int
    cost: float

    def isArrived(self, step: int) -> bool:
        """
        Return True if the shipment has reached its destination by the given time step.

        This supports a clean "receive shipments" phase each step.
        """
        return step >= self.arrivalStep


@dataclass
class DisruptionScenario:
    """
    Represents a disruption that affects the network during a specific time window.

    disruptionType options in this baseline:
    supplier_shutdown, lane_delay, lane_capacity_drop, demand_spike
    """
    scenarioId: str
    disruptionType: str
    startStep: int
    endStep: int
    targetId: str  # node id or "SRC->DST"
    severity: float

    @property
    def duration(self) -> int:
        """
        Duration in time steps.

        Duration is a useful derived feature for reporting and ML.
        """
        return int(self.endStep - self.startStep + 1)


@dataclass
class SupplyChainModel:
    """
    Aggregates network structure and state.

    Owns:
    nodes, lanes, and shipments currently in transit.
    """
    nodes: Dict[NodeId, SupplyNode] = field(default_factory=dict)
    lanes: Dict[EdgeId, TransportationLane] = field(default_factory=dict)
    inTransit: List[Shipment] = field(default_factory=list)

    def addNode(self, node: SupplyNode) -> None:
        """
        Add a node to the network.

        Storing nodes by id ensures stable references during simulation.
        """
        self.nodes[node.node_id] = node

    def addLane(self, lane: TransportationLane) -> None:
        """
        Add a lane to the network keyed by (source, destination).

        Tuple keys provide fast lookup for shipping and disruptions.
        """
        self.lanes[(lane.sourceNode, lane.destinationNode)] = lane

    @staticmethod
    def laneId(src: NodeId, dst: NodeId) -> str:
        """
        Create a human readable lane id such as P1->D1.

        This is convenient for scenario configuration and persistence.
        """
        return f"{src}->{dst}"

    @staticmethod
    def parseLaneId(lane_id: str) -> EdgeId:
        """
        Convert a string lane id like P1->D1 into a tuple key (P1, D1).

        Disruptions reference lanes using strings, then parse them for lookup.
        """
        parts = lane_id.split("->")
        if len(parts) != 2:
            raise ValueError(f"Invalid lane id: {lane_id}")
        return (parts[0], parts[1])

    def getLane(self, src: NodeId, dst: NodeId) -> TransportationLane:
        """
        Retrieve a lane by its tuple key.

        This keeps lane access consistent and avoids repeated indexing patterns.
        """
        return self.lanes[(src, dst)]

    def incomingSources(self, dst: NodeId) -> List[NodeId]:
        """
        Return all upstream sources that have lanes into dst.

        Replenishment ordering requires knowledge of valid upstream sources.
        """
        return [s for (s, d) in self.lanes.keys() if d == dst]

    def retailNodes(self) -> List[NodeId]:
        """
        Return ids of retail nodes.

        Retail nodes generate demand and accumulate backlog.
        """
        return [n.node_id for n in self.nodes.values() if n.node_type == "retail"]

    def plantNodes(self) -> List[NodeId]:
        """
        Return ids of plant nodes.

        Plant nodes produce inventory each time step based on capacity.
        """
        return [n.node_id for n in self.nodes.values() if n.node_type == "plant"]


@dataclass
class Metrics:
    """
    Run summary metrics.

    These metrics support baseline vs disrupted comparisons and ML dataset creation.
    """
    serviceLevel: float
    stockoutSteps: int
    totalCost: float
    transportCost: float
    holdingCost: float
    shortageCost: float
    recoveryTime: int
    demanded: float
    fulfilled: float


def default_demand(step: int, base: float, noise: float, seasonal: float = 0.0) -> float:
    """
    Generate synthetic retail demand for one time step.

    base sets expected demand.
    noise adds variability.
    seasonal adds an optional periodic pattern.
    """
    season = seasonal * math.sin(2 * math.pi * (step % 14) / 14)
    return max(0.0, base + season + random.uniform(-noise, noise))


class SimulationController:
    """
    Discrete time simulation engine.

    Advances the model through time steps and returns run metrics.
    """
    def __init__(
        self,
        model: SupplyChainModel,
        horizon: int,
        demand_base: Dict[NodeId, float],
        demand_noise: float = 4.0,
        demand_seasonal: float = 0.0,
        seed: int = 7,
    ):
        """
        Store run settings and initialize counters.

        A fixed seed enables reproducible runs for debugging and evaluation.
        """
        self.model = model
        self.horizon = horizon
        self.demand_base = demand_base
        self.demand_noise = demand_noise
        self.demand_seasonal = demand_seasonal
        self.seed = seed

        self.transportCost = 0.0
        self.holdingCost = 0.0
        self.shortageCost = 0.0
        self.demanded = 0.0
        self.fulfilled = 0.0
        self.stockoutSteps = 0

        random.seed(seed)

    def applyDisruption(self, step: int, scenario: Optional[DisruptionScenario]) -> None:
        """
        Apply disruption effects if scenario is active at the given step.

        supplier_shutdown reduces node capacity.
        lane_delay adds delay steps to a lane.
        lane_capacity_drop reduces lane capacity.
        demand_spike is handled during demand generation.
        """
        if scenario is None:
            return
        if not (scenario.startStep <= step <= scenario.endStep):
            return

        t = scenario.disruptionType
        if t == "supplier_shutdown":
            node = self.model.nodes.get(scenario.targetId)
            if node is not None:
                node.capacity = node.capacity * scenario.severity
        elif t == "lane_delay":
            eid = self.model.parseLaneId(scenario.targetId)
            lane = self.model.lanes.get(eid)
            if lane is not None:
                lane.delaySteps = int(scenario.severity)
        elif t == "lane_capacity_drop":
            eid = self.model.parseLaneId(scenario.targetId)
            lane = self.model.lanes.get(eid)
            if lane is not None:
                lane.capacity = lane.capacity * scenario.severity
        elif t == "demand_spike":
            return
        else:
            raise ValueError(f"Unknown disruptionType: {t}")

    def _resetTransientDisruptionState(self, scenario: Optional[DisruptionScenario]) -> None:
        """
        Reset disruption modifiers that should not persist outside the disruption window.

        Lane delays are temporary modifiers, so delaySteps is reset each step before reapplying.
        """
        if scenario is None:
            return
        if scenario.disruptionType == "lane_delay":
            eid = self.model.parseLaneId(scenario.targetId)
            lane = self.model.lanes.get(eid)
            if lane is not None:
                lane.delaySteps = 0

    def _receiveShipments(self, step: int) -> None:
        """
        Move arrived shipments into destination inventory and accumulate transport cost.

        This enforces lane lead times by delaying availability until arrival.
        """
        arrived = [s for s in self.model.inTransit if s.isArrived(step)]
        self.model.inTransit = [s for s in self.model.inTransit if not s.isArrived(step)]
        for s in arrived:
            self.model.nodes[s.destinationNode].updateInventory(s.qty)
            self.transportCost += s.cost

    def _produce(self) -> None:
        """
        Add production output to plant inventories based on capacity.

        This baseline model treats production as independent of upstream raw materials.
        """
        for pid in self.model.plantNodes():
            plant = self.model.nodes[pid]
            produced = max(0.0, plant.capacity)
            plant.updateInventory(produced)

    def _fillBacklog(self, retail_id: NodeId) -> None:
        """
        Fulfill existing backlog at a retail node using available inventory.

        Processing backlog first models accumulation of unmet demand across steps.
        """
        node = self.model.nodes[retail_id]
        if node.backlog <= 0:
            return
        filled = min(node.inventory_level, node.backlog)
        node.inventory_level -= filled
        node.backlog -= filled
        self.fulfilled += filled

    def _consumeDemand(self, step: int, scenario: Optional[DisruptionScenario]) -> None:
        """
        Generate demand, fulfill it, and push any unmet amount into backlog.

        This drives service level and stockout metrics.
        """
        for rid in self.model.retailNodes():
            base = self.demand_base.get(rid, 30.0)
            d = default_demand(step, base, self.demand_noise, self.demand_seasonal)

            if (
                scenario is not None
                and scenario.disruptionType == "demand_spike"
                and scenario.targetId == rid
                and scenario.startStep <= step <= scenario.endStep
            ):
                d = d * scenario.severity

            self.demanded += d
            node = self.model.nodes[rid]

            filled = min(node.inventory_level, d)
            node.inventory_level -= filled
            self.fulfilled += filled

            remaining = d - filled
            if remaining > 0:
                node.backlog += remaining
                self.stockoutSteps += 1

    def _placeReplenishmentOrders(self) -> List[Tuple[NodeId, NodeId, float]]:
        """
        Place replenishment orders using a reorder point, order up to policy.

        Inventory position is defined as inventory minus backlog.
        Orders are split across enabled upstream sources for simplicity in the baseline.
        """
        orders: List[Tuple[NodeId, NodeId, float]] = []
        for node in self.model.nodes.values():
            if node.node_type == "supplier":
                continue

            inventory_position = node.inventory_level - node.backlog
            if inventory_position <= node.reorder_point:
                qty = max(0.0, node.order_up_to - inventory_position)
                if qty <= 0:
                    continue

                srcs = self.model.incomingSources(node.node_id)
                enabled_srcs = [s for s in srcs if self.model.getLane(s, node.node_id).enabled]
                if not enabled_srcs:
                    continue

                split = qty / len(enabled_srcs)
                for s in enabled_srcs:
                    orders.append((s, node.node_id, split))
        return orders

    def _shipOrders(self, step: int, orders: List[Tuple[NodeId, NodeId, float]]) -> None:
        """
        Convert orders into shipments subject to lane capacity and source inventory.

        Constraints here create bottlenecks that propagate disruptions through the network.
        """
        lane_remaining: Dict[EdgeId, float] = {}
        for eid, lane in self.model.lanes.items():
            lane_remaining[eid] = lane.capacity if lane.enabled else 0.0

        for src, dst, requested in orders:
            lane = self.model.lanes.get((src, dst))
            if lane is None or not lane.enabled:
                continue

            cap = lane_remaining[(src, dst)]
            if cap <= 0:
                continue

            src_node = self.model.nodes[src]
            available = src_node.inventory_level
            ship_qty = min(requested, cap, available)
            if ship_qty <= 0:
                continue

            src_node.inventory_level -= ship_qty
            lane_remaining[(src, dst)] -= ship_qty

            shipment = lane.transportGoods(ship_qty, current_step=step)
            self.model.inTransit.append(shipment)

    def _applyCosts(self) -> None:
        """
        Accumulate holding and shortage costs for one time step.

        This enables cost based scenario comparisons in addition to service metrics.
        """
        step_hold = 0.0
        step_short = 0.0
        for node in self.model.nodes.values():
            step_hold += node.calculateStorageCost()
            step_short += node.calculateShortageCost()
        self.holdingCost += step_hold
        self.shortageCost += step_short

    def runSimulation(self, scenario: Optional[DisruptionScenario] = None, target_service: float = 0.95) -> Metrics:
        """
        Execute the simulation and return summary metrics.

        Baseline parameters are restored each step so disruptions remain time bounded.
        Recovery time is measured after disruption ends until service level meets target_service.
        """
        baseline_capacity = {nid: n.capacity for nid, n in self.model.nodes.items()}
        baseline_lane_capacity = {eid: l.capacity for eid, l in self.model.lanes.items()}

        disruption_end = scenario.endStep if scenario else -1
        recovery_time = 0
        recovered = False

        for step in range(self.horizon):
            for nid, cap in baseline_capacity.items():
                self.model.nodes[nid].capacity = cap
            for eid, cap in baseline_lane_capacity.items():
                self.model.lanes[eid].capacity = cap
            self._resetTransientDisruptionState(scenario)

            self.applyDisruption(step, scenario)

            self._receiveShipments(step)
            self._produce()

            for rid in self.model.retailNodes():
                self._fillBacklog(rid)

            self._consumeDemand(step, scenario)

            orders = self._placeReplenishmentOrders()
            self._shipOrders(step, orders)

            self._applyCosts()

            sl = (self.fulfilled / self.demanded) if self.demanded > 0 else 0.0
            if scenario and step > disruption_end and not recovered:
                if sl >= target_service:
                    recovered = True
                    recovery_time = step - disruption_end

        service_level = (self.fulfilled / self.demanded) if self.demanded > 0 else 0.0
        total_cost = self.transportCost + self.holdingCost + self.shortageCost

        return Metrics(
            serviceLevel=service_level,
            stockoutSteps=self.stockoutSteps,
            totalCost=total_cost,
            transportCost=self.transportCost,
            holdingCost=self.holdingCost,
            shortageCost=self.shortageCost,
            recoveryTime=recovery_time if scenario else 0,
            demanded=self.demanded,
            fulfilled=self.fulfilled,
        )


class VisualizationManager:
    """
    Placeholder visualization layer.

    A later implementation could export charts to files or provide a dashboard interface.
    """
    def generateGraphs(self) -> None:
        return

    def displayMetrics(self) -> None:
        return