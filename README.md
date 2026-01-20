# 二维伊辛模型（蒙特卡洛）

本脚本使用 Metropolis 算法模拟 100x100 的二维伊辛模型，
采用周期性边界条件，温度为约化温度 `T = kB T / J`。
这里的 `mc-steps` 表示“扫场次数”（每次扫场包含 `N=L*L` 次随机翻转尝试）。

## 使用方法

```bash
python3 main.py
```

## 冷启动
```shell
# 从 T=1.0 开始，步长 0.05，初始全向上，启用状态延续(anneal)
python3 main.py --L 100 --J 1.0 --T-start 1.0 --T-stop 3.5 --T-step 0.05 --mc-steps 5000 --measure-interval 5 --init up --anneal
```

## 精细
```shell
python3 main.py --T-start 2.0 --T-stop 2.6 --T-step 0.02 --mc-steps 10000 --init random
```