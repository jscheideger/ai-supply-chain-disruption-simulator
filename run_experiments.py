from __future__ import annotations

# copy supports deep copying mutable network state between baseline and scenario runs.
import copy

# typing documents expected data structures used during dataset generation.
from typing import Dict, List

# pandas stores the dataset and supports export to CSV.
import pandas as pd

from simulation import SupplyChainModel, SimulationController
from synth import build_tiered_network, graph_features, random_scenario
from ml import MachineLearningModel


def clone_model(model: SupplyChainModel) -> SupplyChainModel:
    """
    Return a deep copy of the supply chain model.

    The simulator mutates inventories, backlogs, and shipments.
    Deep copying ensures baseline and scenario runs start from identical initial conditions.
    """
    return copy.deepcopy(model)


def build_dataset(
    n_runs: int = 450,
    horizon: int = 90,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a dataset for ML training.

    Each row represents a single disruption scenario run with:
    network features, scenario parameters, baseline metrics, and scenario outcome metrics.
    """
    rows: List[Dict] = []

    for i in range(n_runs):
        # Seed schedule creates run to run variation while remaining reproducible.
        net_seed = seed + i * 13

        # Build a synthetic network and its demand profile.
        model, demand_base = build_tiered_network(
            seed=net_seed,
            suppliers=1,
            plants=1,
            dcs=1,
            retails=1,
        )

        # Graph features describe network topology and average lane properties.
        feats = graph_features(model)

        # Baseline run: no disruption.
        base_model = clone_model(model)
        sim_base = SimulationController(
            base_model,
            horizon=horizon,
            demand_base=demand_base,
            seed=net_seed,
        )
        m_base = sim_base.runSimulation(scenario=None)

        # Disrupted run: a randomized scenario is applied.
        scenario = random_scenario(model, horizon=horizon, seed=net_seed + 3)
        sc_model = clone_model(model)
        sim_sc = SimulationController(
            sc_model,
            horizon=horizon,
            demand_base=demand_base,
            seed=net_seed,
        )
        m_sc = sim_sc.runSimulation(scenario=scenario)

        # One dataset row is created by combining inputs and outputs.
        row = {
            **feats,
            "kind": scenario.disruptionType,
            "severity": scenario.severity,
            "start_step": scenario.startStep,
            "duration": scenario.duration,
            "baseline_service_level": m_base.serviceLevel,
            "baseline_total_cost": m_base.totalCost,
            "baseline_stockout_steps": m_base.stockoutSteps,
            "scenario_service_level": m_sc.serviceLevel,
            "scenario_total_cost": m_sc.totalCost,
            "scenario_stockout_steps": m_sc.stockoutSteps,
            "scenario_recovery_time": m_sc.recoveryTime,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """
    Orchestrate an end to end run.

    Produces:
    simulation_training_data.csv
    ML model artifacts in artifacts/
    printed training metrics for reporting
    """
    df = build_dataset(n_runs=450, horizon=90, seed=7)

    # Saving the dataset provides reproducible evidence and supports later analysis.
    df.to_csv("simulation_training_data.csv", index=False)

    # Train the ML models and save them for later reuse.
    ml = MachineLearningModel()
    metrics = ml.trainModel(df, out_dir="artifacts")

    print("Training metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()