# 二维伊辛模型（蒙特卡洛）

本脚本使用 Metropolis 算法模拟 100x100 的二维伊辛模型，
采用周期性边界条件，温度为约化温度 `T = kB T / J`。
这里的 `mc-steps` 表示“扫场次数”（每次扫场包含 `N=L*L` 次随机翻转尝试）。

## 使用方法

```bash
python3 main.py
```

常用参数：

```bash
python3 main.py --L 100 --mc-steps 10000 --T-start 0.2 --T-stop 6.0 --T-step 0.2
```

结果会输出到控制台，并保存到 `ising_results.csv`，列为：
`T`、`M_abs_per_spin`、`E_per_spin`、`Cv_per_spin`。绘图文件为：
`ising_M_vs_T.png`、`ising_E_vs_T.png`、`ising_Cv_vs_T.png`。
