from __future__ import annotations

# typing documents return types and expected data structures.
from typing import Dict, List, Tuple

# random generates synthetic parameters and scenarios with reproducibility via seeding.
import random

# networkx is used to compute basic topology features for ML.
import networkx as nx

from simulation import SupplyChainModel, SupplyNode, TransportationLane, DisruptionScenario


def build_tiered_network(
    seed: int = 7,
    suppliers: int = 1,
    plants: int = 1,
    dcs: int = 1,
    retails: int = 1,
) -> Tuple[SupplyChainModel, Dict[str, float]]:
    """
    Build a tiered network S -> P -> D -> R with randomized but reasonable parameters.

    This acts as a synthetic data generator for:
    simulation testing, scenario experiments, and ML dataset creation.
    """
    random.seed(seed)
    model = SupplyChainModel()
    demand_base: Dict[str, float] = {}

    def add_nodes(prefix: str, count: int, node_type: str) -> List[str]:
        """
        Create multiple nodes of a given type.

        Parameter ranges are chosen to keep the baseline stable while still allowing disruption effects.
        """
        ids: List[str] = []
        for i in range(1, count + 1):
            nid = f"{prefix}{i}"

            if node_type == "supplier":
                cap = 9999.0
                inv = random.uniform(3000, 7000)
                hold = 0.005
                short = 0.0
                rp = 0.0
                out_to = 0.0
            elif node_type == "plant":
                cap = random.uniform(60, 120)
                inv = random.uniform(50, 150)
                hold = 0.02
                short = 0.0
                rp = 80.0
                out_to = 240.0
            elif node_type == "dc":
                cap = 0.0
                inv = random.uniform(80, 220)
                hold = 0.015
                short = 0.0
                rp = 90.0
                out_to = 260.0
            else:
                cap = 0.0
                inv = random.uniform(60, 140)
                hold = 0.01
                short = 0.25
                rp = 70.0
                out_to = 210.0
                demand_base[nid] = random.uniform(25, 45)

            model.addNode(
                SupplyNode(
                    node_id=nid,
                    node_type=node_type,
                    capacity=cap,
                    holding_cost_per_unit=hold,
                    shortage_cost_per_unit=short,
                    reorder_point=rp,
                    order_up_to=out_to,
                    inventory_level=inv,
                    backlog=0.0,
                )
            )
            ids.append(nid)
        return ids

    suppliers_ids = add_nodes("S", suppliers, "supplier")
    plants_ids = add_nodes("P", plants, "plant")
    dcs_ids = add_nodes("D", dcs, "dc")
    retails_ids = add_nodes("R", retails, "retail")

    def connect_many(srcs: List[str], dsts: List[str]) -> None:
        """
        Connect every node in srcs to every node in dsts.

        Full connectivity within a tier provides multiple routing options and avoids dead ends.
        """
        for s in srcs:
            for d in dsts:
                model.addLane(
                    TransportationLane(
                        sourceNode=s,
                        destinationNode=d,
                        capacity=random.uniform(80, 220),
                        leadTime=random.randint(1, 3),
                        costPerUnit=random.uniform(0.04, 0.14),
                        enabled=True,
                        delaySteps=0,
                    )
                )

    connect_many(suppliers_ids, plants_ids)
    connect_many(plants_ids, dcs_ids)
    connect_many(dcs_ids, retails_ids)

    return model, demand_base


def graph_features(model: SupplyChainModel) -> Dict[str, float]:
    """
    Compute graph level features that help ML models learn network resilience patterns.

    Features include:
    node count, edge count, average degrees, average lead time, and average lane capacity.
    """
    g = nx.DiGraph()
    for nid in model.nodes.keys():
        g.add_node(nid)
    for (s, d), lane in model.lanes.items():
        if lane.enabled:
            g.add_edge(s, d)

    n_nodes = float(g.number_of_nodes())
    n_edges = float(g.number_of_edges())

    avg_out = sum(dict(g.out_degree()).values()) / n_nodes if n_nodes else 0.0
    avg_in = sum(dict(g.in_degree()).values()) / n_nodes if n_nodes else 0.0

    lead_times = [lane.leadTime for lane in model.lanes.values()]
    caps = [lane.capacity for lane in model.lanes.values()]
    avg_lead = sum(lead_times) / len(lead_times) if lead_times else 0.0
    avg_cap = sum(caps) / len(caps) if caps else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_out_degree": float(avg_out),
        "avg_in_degree": float(avg_in),
        "avg_lead_time": float(avg_lead),
        "avg_lane_capacity": float(avg_cap),
    }


def random_scenario(model: SupplyChainModel, horizon: int, seed: int) -> DisruptionScenario:
    """
    Generate a randomized disruption scenario for experiments and ML dataset creation.

    Random selection increases variety across runs, improving training coverage.
    """
    random.seed(seed)
    kind = random.choice(["supplier_shutdown", "lane_delay", "lane_capacity_drop", "demand_spike"])
    start = random.randint(5, max(5, horizon // 2))
    dur = random.randint(3, 12)
    end = min(horizon - 1, start + dur)

    if kind == "supplier_shutdown":
        suppliers_ids = [n.node_id for n in model.nodes.values() if n.node_type == "supplier"]
        target = random.choice(suppliers_ids)
        severity = random.choice([0.0, 0.25, 0.5, 0.75])
    elif kind == "lane_delay":
        (s, d) = random.choice(list(model.lanes.keys()))
        target = f"{s}->{d}"
        severity = float(random.choice([1, 2, 3]))
    elif kind == "lane_capacity_drop":
        (s, d) = random.choice(list(model.lanes.keys()))
        target = f"{s}->{d}"
        severity = random.choice([0.25, 0.5, 0.75])
    else:
        retails_ids = [n.node_id for n in model.nodes.values() if n.node_type == "retail"]
        target = random.choice(retails_ids)
        severity = random.choice([1.2, 1.5, 1.8, 2.2])

    return DisruptionScenario(
        scenarioId=f"sc_{seed}",
        disruptionType=kind,
        startStep=start,
        endStep=end,
        targetId=target,
        severity=float(severity),
    )