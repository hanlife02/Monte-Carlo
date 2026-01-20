#!/usr/bin/env python3
import argparse
import csv
import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def init_spins(L: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice([-1, 1], size=(L, L))


def total_energy(spins: np.ndarray, J: float) -> float:
    # 只统计一次相互作用（向右与向下的近邻）。
    right = np.roll(spins, shift=-1, axis=1)
    down = np.roll(spins, shift=-1, axis=0)
    return -J * np.sum(spins * (right + down))


def simulate(
    L: int,
    J: float,
    T: float,
    total_steps: int,
    thermal_steps: int,
    measure_interval: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    spins = init_spins(L, rng)
    N = L * L
    E = total_energy(spins, J)
    M = int(np.sum(spins))

    sum_E = 0.0
    sum_E2 = 0.0
    sum_Mabs = 0.0
    n_meas = 0

    for step in range(total_steps):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        s = spins[i, j]
        neighbor_sum = (
            spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
        )
        dE = 2.0 * J * s * neighbor_sum
        if dE <= 0.0 or rng.random() < math.exp(-dE / T):
            spins[i, j] = -s
            E += dE
            M -= 2 * s

        if step >= thermal_steps and (step - thermal_steps + 1) % measure_interval == 0:
            n_meas += 1
            sum_E += E
            sum_E2 += E * E
            sum_Mabs += abs(M)

    if n_meas == 0:
        raise RuntimeError("No measurements collected; increase total_steps or decrease thermal_steps.")

    avg_E = sum_E / n_meas
    avg_E2 = sum_E2 / n_meas
    avg_Mabs = sum_Mabs / n_meas

    E_per_spin = avg_E / N
    Mabs_per_spin = avg_Mabs / N
    Cv_per_spin = (avg_E2 - avg_E * avg_E) / (N * T * T)

    return Mabs_per_spin, E_per_spin, Cv_per_spin


def estimate_tc(temps: np.ndarray, cvs: np.ndarray) -> float:
    # 以热容峰值对应温度作为相变温度的估计。
    idx = int(np.argmax(cvs))
    return float(temps[idx])


def run_simulation(args: argparse.Namespace) -> float:
    N = args.L * args.L
    thermal_steps = args.thermal_steps
    if thermal_steps is None:
        thermal_steps = args.mc_steps // 5

    measure_interval = args.measure_interval
    if measure_interval is None:
        measure_interval = N

    rng = np.random.default_rng(args.seed)
    temps = np.arange(args.T_start, args.T_stop + 1e-12, args.T_step)
    temp_list = []
    cv_list = []

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "M_abs_per_spin", "E_per_spin", "Cv_per_spin"])
        for T in temps:
            m, e, cv = simulate(
                L=args.L,
                J=args.J,
                T=float(T),
                total_steps=args.mc_steps,
                thermal_steps=thermal_steps,
                measure_interval=measure_interval,
                rng=rng,
            )
            writer.writerow([f"{T:.6g}", f"{m:.6g}", f"{e:.6g}", f"{cv:.6g}"])
            print(f"T={T:.3f}  <|M|>/N={m:.6f}  <E>/N={e:.6f}  Cv/N={cv:.6f}")
            temp_list.append(float(T))
            cv_list.append(cv)

    if not cv_list:
        raise RuntimeError("No Cv data collected.")
    return estimate_tc(np.array(temp_list), np.array(cv_list))


def plot_results(csv_path: str, prefix: str) -> float:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    T = data["T"]
    M = data["M_abs_per_spin"]
    E = data["E_per_spin"]
    Cv = data["Cv_per_spin"]

    plt.figure()
    plt.plot(T, M, "o-", lw=1)
    plt.xlabel("T (kB T / J)")
    plt.ylabel("<|M|>/N")
    plt.title("Magnetization vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_M_vs_T.png", dpi=150)

    plt.figure()
    plt.plot(T, E, "o-", lw=1)
    plt.xlabel("T (kB T / J)")
    plt.ylabel("<E>/N")
    plt.title("Energy vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_E_vs_T.png", dpi=150)

    plt.figure()
    plt.plot(T, Cv, "o-", lw=1)
    plt.xlabel("T (kB T / J)")
    plt.ylabel("Cv/N")
    plt.title("Heat Capacity vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_Cv_vs_T.png", dpi=150)
    return estimate_tc(T, Cv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Ising model Monte Carlo (Metropolis)")
    parser.add_argument("--L", type=int, default=100, help="Linear system size")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant J")
    parser.add_argument("--T-start", type=float, default=0.2, help="Starting reduced temperature")
    parser.add_argument("--T-stop", type=float, default=6.0, help="Ending reduced temperature (inclusive)")
    parser.add_argument("--T-step", type=float, default=0.2, help="Temperature step")
    parser.add_argument("--mc-steps", type=int, default=1_000_000, help="Total MC steps (spin-flip attempts)")
    parser.add_argument(
        "--thermal-steps",
        type=int,
        default=None,
        help="Thermalization steps (defaults to 20% of mc-steps)",
    )
    parser.add_argument(
        "--measure-interval",
        type=int,
        default=None,
        help="Measurement interval in steps (defaults to one sweep)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="ising_results.csv",
        help="CSV output file",
    )
    parser.add_argument("--prefix", type=str, default="ising", help="Image output prefix")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.L <= 0:
        raise ValueError("L must be positive")
    if args.mc_steps <= 0:
        raise ValueError("mc-steps must be positive")
    if args.T_step <= 0:
        raise ValueError("T-step must be positive")

    if args.plot_only:
        tc = plot_results(args.output, args.prefix)
        print(f"估计相变温度 Tc ≈ {tc:.3f}（由 Cv 峰值确定）")
        return

    tc = run_simulation(args)
    if not args.no_plot:
        plot_results(args.output, args.prefix)
    print(f"估计相变温度 Tc ≈ {tc:.3f}（由 Cv 峰值确定）")


if __name__ == "__main__":
    main()
