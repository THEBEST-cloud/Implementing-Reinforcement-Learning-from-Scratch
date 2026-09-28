# Policy Gradient 学习笔记

入口：[`../02-Policy-Gradient.ipynb`](../02-Policy-Gradient.ipynb)。参考 OpenAI Spinning Up，包含整局回报、Reward-to-Go 与价值基线三个教学实现。

## 运行

使用现有 `rl_env`，在项目根目录打开 Notebook，并选择该环境的 Python 内核，从上到下运行。已保存的输出可直接阅读；代码不调用在线模型，也不下载数据。

```bash
conda activate rl_env
python -c "import sys; print(sys.executable)"
jupyter notebook 02-Policy-Gradient.ipynb
```

若编辑了生成脚本，可以在项目根目录重新生成并执行：

```bash
python scripts/build_policy_gradient_notebook.py --execute
```

此命令会覆盖第二章 Notebook 和本章实验产物；直接编辑 Notebook 后，不要运行生成脚本，否则手工编辑会被覆盖。第一章不受影响。`requirements.txt` 记录本次执行所用版本，跨平台执行不保证数值逐位一致；绘图还需要系统安装中文字体，例如 Noto Sans CJK SC。

## 实验产物

- `artifacts/01_*.png` 至 `10_*.png`：笔记中的十张代码生成图。
- `artifacts/training.csv`：三种方法、三个训练种子、每个种子 60 批训练记录。
- `artifacts/evaluation.csv`：各最终策略的独立评估，分别报告随机采样和 argmax。
- `artifacts/run_config.json`：运行版本、种子和超参数。
- `artifacts/rollout.npz`：固定策略与固定种子的真实轨迹。

所有方法使用相同训练回合数，并非相同环境交互步数。只有三个训练种子，结果用于学习和复现，不构成普遍的算法排名。
