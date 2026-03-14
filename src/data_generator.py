"""
Data Generator for Project 003: Tree-Based Learning
Generates two synthetic datasets:
  A) EV Battery Thermal Runaway Prediction (general, intuitive)
  B) Wafer Edge Yield Drop-off (post-silicon validation)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_ev_battery_data(n=8000, seed=42):
    """
    EV Battery Thermal Runaway Prediction
    Target: thermal_runaway (1=Fire, 0=Safe)  ~5% positive class
    Features: charge_rate_kw, ambient_temp_c, cooling_flow_rate,
              cell_voltage_variance, age_cycles
    """
    rng = np.random.default_rng(seed)
    n_fire = int(n * 0.05)
    n_safe = n - n_fire

    # Safe batteries: moderate charge, normal temp, good cooling
    safe = pd.DataFrame({
        "charge_rate_kw":        rng.normal(7.0, 1.5, n_safe).clip(1, 22),
        "ambient_temp_c":        rng.normal(22.0, 6.0, n_safe).clip(-10, 45),
        "cooling_flow_rate":     rng.normal(5.0, 1.0, n_safe).clip(1, 10),
        "cell_voltage_variance": rng.exponential(0.005, n_safe).clip(0, 0.05),
        "age_cycles":            rng.integers(0, 1000, n_safe),
        "thermal_runaway":       0
    })

    # Fire batteries: high charge, high temp, poor cooling, high variance, old
    fire = pd.DataFrame({
        "charge_rate_kw":        rng.normal(18.0, 2.5, n_fire).clip(10, 22),
        "ambient_temp_c":        rng.normal(42.0, 4.0, n_fire).clip(30, 55),
        "cooling_flow_rate":     rng.normal(1.5, 0.5, n_fire).clip(0.1, 3),
        "cell_voltage_variance": rng.exponential(0.04, n_fire).clip(0.02, 0.2),
        "age_cycles":            rng.integers(600, 1500, n_fire),
        "thermal_runaway":       1
    })

    df = pd.concat([safe, fire], ignore_index=True).sample(frac=1, random_state=seed)
    df.to_csv(DATA_DIR / "ev_battery_thermal_runaway.csv", index=False)
    print(f"EV Battery dataset: {len(df)} rows, {df['thermal_runaway'].mean():.1%} fire rate")
    return df


def generate_wafer_yield_data(n=10000, seed=99):
    """
    Wafer Edge Yield Drop-off
    Target: yield_pass (1=Pass, 0=Fail)  ~85% pass rate
    Features: distance_from_center_mm, angle_degrees, etch_gas_flow,
              spin_coat_rpm, litho_overlay_error_nm
    """
    rng = np.random.default_rng(seed)

    distance   = rng.uniform(0, 150, n)       # 0 = center, 150 = edge
    angle      = rng.uniform(0, 360, n)
    etch_flow  = rng.normal(50, 5, n).clip(30, 70)
    spin_rpm   = rng.normal(2000, 150, n).clip(1500, 2500)
    overlay_nm = rng.normal(2.0, 0.8, n).clip(0, 8)

    # Yield drops non-linearly near the edge (distance > 120mm)
    edge_penalty = np.where(distance > 120, (distance - 120) / 30, 0)
    overlay_penalty = np.where(overlay_nm > 4, (overlay_nm - 4) / 4, 0)
    etch_penalty = np.where(np.abs(etch_flow - 50) > 10, 0.2, 0)

    fail_prob = 0.05 + 0.5 * edge_penalty + 0.3 * overlay_penalty + 0.1 * etch_penalty
    fail_prob = fail_prob.clip(0, 0.95)
    yield_pass = rng.binomial(1, 1 - fail_prob, n)

    df = pd.DataFrame({
        "distance_from_center_mm": distance.round(2),
        "angle_degrees":           angle.round(2),
        "etch_gas_flow":           etch_flow.round(2),
        "spin_coat_rpm":           spin_rpm.round(0).astype(int),
        "litho_overlay_error_nm":  overlay_nm.round(3),
        "yield_pass":              yield_pass
    })

    df.to_csv(DATA_DIR / "wafer_edge_yield.csv", index=False)
    print(f"Wafer Yield dataset: {len(df)} rows, {df['yield_pass'].mean():.1%} pass rate")
    return df


if __name__ == "__main__":
    generate_ev_battery_data()
    generate_wafer_yield_data()
    print("Both datasets generated successfully.")
