# 手撕强化学习

从 Q 表到策略梯度，用中文学习笔记、数字手算、可运行代码和可视化理解强化学习。理论学习参考 OpenAI Spinning Up；本项目是独立学习实现，不是 OpenAI 官方项目或官方基准复现。

## 学习顺序

1. [Q-Table：从零理解 Q-learning](01Q-Table.ipynb)：在 FrozenLake 中理解状态、动作、奖励、探索与 TD 更新。
2. [Policy Gradient：从 Q 表到动作概率](02-Policy-Gradient.ipynb)：在 CartPole 中实现整局回报、Reward-to-Go 与价值基线，包含手算、梯度检查、多种子实验和独立评估。

Notebook 已保存图文和运行输出，可以直接阅读。建议从上到下逐个运行单元格，再修改参数验证自己的理解。

## 运行环境

当前验证环境为 Python 3.10 的 `rl_env`。在仓库根目录启动 Jupyter，选择该环境对应的内核：

```bash
conda activate rl_env
python -m pip install -r policy_gradient/requirements.txt
python -m notebook
```

如果环境尚未安装 Jupyter Notebook，可先运行 `python -m pip install notebook`。已有依赖齐全时无需重复安装。依赖文件记录实验所用版本，跨平台运行不保证数值逐位一致。

绘图需要中文字体，例如 Noto Sans CJK SC、Hiragino Sans GB 或微软雅黑。安装字体后请重启内核；第二章会检查可用字体，避免生成乱码。

## 目录

```text
01Q-Table.ipynb                     Q-learning 学习笔记
02-Policy-Gradient.ipynb            策略梯度学习笔记
assets/q_learning_cover.png         第一章插图
scripts/build_policy_gradient_notebook.py
                                   第二章 Notebook 生成脚本
policy_gradient/
  README.md                        第二章运行与复现说明
  requirements.txt                 已验证的依赖版本
  artifacts/                       实验配置、训练记录、评估数据和图表
```

本仓库仅跟踪学习笔记、算法代码及相关实验资料，不包含视频制作工具或素材。

## 第二章复现

直接运行 Notebook 即可更新本章实验产物。也可以通过脚本重新生成并执行：

```bash
python scripts/build_policy_gradient_notebook.py --execute
```

**注意：生成脚本会覆盖第二章 Notebook 及实验产物。** 若直接修改了 Notebook，请先保存自己的修改；不带 `--execute` 运行会生成未执行的 Notebook，不保留已有输出。

实验使用三个预先固定的训练种子；各方法的训练回合数相同，但环境交互步数并不相同。评估分别报告随机采样和 argmax，结果用于学习，不构成普遍的算法排名。详见[第二章说明](policy_gradient/README.md)。

## 参考资料

- [OpenAI Spinning Up：强化学习核心概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [OpenAI Spinning Up：策略优化入门](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Gymnasium 文档](https://gymnasium.farama.org/)

各章还包含与具体推导、环境和实现对应的参考链接。
