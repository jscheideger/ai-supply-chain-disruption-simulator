from __future__ import annotations

# copy ensures the baseline and scenario runs start with identical initial conditions.
import copy

from simulation import SimulationController, DisruptionScenario
from synth import build_tiered_network


def main() -> None:
    """
    Run a small sanity check:
    1 baseline simulation
    2 a single disruption simulation
    Then print both metric objects.

    This provides a quick confirmation that disruptions meaningfully affect outcomes.
    """
    horizon = 90
    seed = 7

    model, demand_base = build_tiered_network(seed=seed, suppliers=1, plants=1, dcs=1, retails=1)

    base_model = copy.deepcopy(model)
    base = SimulationController(base_model, horizon=horizon, demand_base=demand_base, seed=seed)
    baseline_metrics = base.runSimulation(None)

    # Example disruption: lane delay on P1->D1.
    lane_id = "P1->D1"
    scenario = DisruptionScenario(
        scenarioId="demo_delay",
        disruptionType="lane_delay",
        startStep=40,
        endStep=55,
        targetId=lane_id,
        severity=2.0,
    )

    sc_model = copy.deepcopy(model)
    sim = SimulationController(sc_model, horizon=horizon, demand_base=demand_base, seed=seed)
    scenario_metrics = sim.runSimulation(scenario)

    print("Baseline metrics")
    print(baseline_metrics)
    print()
    print("Scenario metrics")
    print(scenario_metrics)


if __name__ == "__main__":
    main()