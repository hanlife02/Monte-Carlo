#!/usr/bin/env python3
import argparse
import csv
import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def init_spins(L: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "random":
        return rng.choice([-1, 1], size=(L, L))
    if mode == "up":
        return np.ones((L, L), dtype=int)
    if mode == "down":
        return -np.ones((L, L), dtype=int)
    raise ValueError(f"Unsupported init mode: {mode}")


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
    spins: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    N = L * L
    E = total_energy(spins, J)
    M = int(np.sum(spins))

    sum_E = 0.0
    sum_E2 = 0.0
    sum_Mabs = 0.0
    n_meas = 0

    for sweep in range(total_steps):
        # 一次扫场：随机尝试翻转 N 次
        for _ in range(N):
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

        if sweep >= thermal_steps and (sweep - thermal_steps + 1) % measure_interval == 0:
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

    return Mabs_per_spin, E_per_spin, Cv_per_spin, spins


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
        measure_interval = 1
    if measure_interval <= 0:
        raise ValueError("measure-interval must be positive")

    rng = np.random.default_rng(args.seed)
    temps = np.arange(args.T_start, args.T_stop + 1e-12, args.T_step)
    temp_list = []
    cv_list = []
    spins = None

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "M_abs_per_spin", "E_per_spin", "Cv_per_spin"])
        for T in temps:
            if args.anneal and spins is None:
                spins = init_spins(args.L, rng, args.init)
            elif not args.anneal:
                spins = init_spins(args.L, rng, args.init)

            m, e, cv, spins = simulate(
                L=args.L,
                J=args.J,
                T=float(T),
                total_steps=args.mc_steps,
                thermal_steps=thermal_steps,
                measure_interval=measure_interval,
                rng=rng,
                spins=spins,
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
    parser = argparse.ArgumentParser(description="二维伊辛模型蒙特卡洛（Metropolis）")
    parser.add_argument("--L", type=int, default=100, help="线性尺寸 L")
    parser.add_argument("--J", type=float, default=1.0, help="耦合常数 J")
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        choices=["random", "up", "down"],
        help="初始自旋构型：random/up/down",
    )
    parser.add_argument("--T-start", type=float, default=0.2, help="起始约化温度")
    parser.add_argument("--T-stop", type=float, default=6.0, help="终止约化温度（包含）")
    parser.add_argument("--T-step", type=float, default=0.2, help="温度步长")
    parser.add_argument(
        "--mc-steps",
        type=int,
        default=100,
        help="MC 扫场次数（每次扫场尝试 N 次翻转）",
    )
    parser.add_argument(
        "--thermal-steps",
        type=int,
        default=None,
        help="热化扫场数（默认 mc-steps 的 20%）",
    )
    parser.add_argument(
        "--measure-interval",
        type=int,
        default=None,
        help="测量间隔（扫场数，默认 1）",
    )
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument(
        "--output",
        type=str,
        default="ising_results.csv",
        help="CSV 输出文件",
    )
    parser.add_argument("--prefix", type=str, default="ising", help="图片输出前缀")
    parser.add_argument("--no-plot", action="store_true", help="仅模拟不绘图")
    parser.add_argument("--plot-only", action="store_true", help="仅根据已有 CSV 绘图")
    parser.add_argument(
        "--anneal",
        action="store_true",
        help="相邻温度点复用最终构型作为初始态",
    )
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
