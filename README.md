# CardioSim

**Gymnasium environments for reinforcement learning in cardiac electrophysiology**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-134%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/cardiosim.svg)](https://pypi.org/project/cardiosim/)

---

CardioSim provides three Gymnasium-compatible reinforcement learning environments for cardiac electrophysiology: pacemaker control, antiarrhythmic drug dosing, and defibrillation timing. It includes the FitzHugh-Nagumo and Aliev-Panfilov cardiac action potential models, a single-compartment pharmacokinetic model, a cardiac conduction system model, configurable difficulty tiers, and baseline agents (random, heuristic, PPO). The package is designed for benchmarking RL algorithms on sequential decision-making problems in cardiology where real-world experimentation is impractical.

## Installation

```bash
pip install cardiosim              # Core (numpy, scipy, gymnasium)
pip install cardiosim[train]       # + SB3, PyTorch for RL training
pip install cardiosim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/cardiosim.git
cd cardiosim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import cardiosim

env = gym.make("cardiosim/PacingControl-v0")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environments

| Environment | Paradigm | Observation | Action | Key Challenge |
|---|---|---|---|---|
| `PacingControl-v0` | Pacemaker optimization | Voltage, recovery, HR, HR error, HRV | Pacing rate, amplitude | Restore normal sinus rhythm in bradycardia |
| `AntiarrhythmicDosing-v0` | Drug dosing | Voltage, recovery, concentration, arrhythmia, efficacy | Drug dose (mg) | Maintain therapeutic window, suppress arrhythmia |
| `DefibrillationTiming-v0` | Defibrillation | Membrane potential, recovery, fibrillation index | Shock decision, energy (J) | Terminate fibrillation with minimal shocks |

## Architecture

CardioSim combines established cardiac electrophysiology models with Gymnasium's RL interface:

- **FitzHugh-Nagumo model** - Two-variable excitable membrane model (FitzHugh, 1961)
- **Aliev-Panfilov model** - Cardiac-specific action potential dynamics (Aliev & Panfilov, 1996)
- **PK/PD model** - Single-compartment pharmacokinetics with sigmoid Emax pharmacodynamics
- **Conduction system** - SA/AV node timing with configurable conduction block
- **Difficulty tiers** - Easy, medium, hard configurations per environment
- **Baseline agents** - Random, heuristic (clinical rules), and PPO (Stable-Baselines3)

## Paper

The accompanying paper is available at:
- [PDF (GitHub)](https://github.com/HassDhia/cardiosim/blob/main/paper/cardiosim.pdf)

## Citation

If you use CardioSim in your research, please cite:

```bibtex
@software{dhia2026cardiosim,
  author = {Dhia, Hass},
  title = {CardioSim: Gymnasium Environments for Reinforcement Learning in Cardiac Electrophysiology},
  year = {2026},
  publisher = {Smart Technology Investments Research Institute},
  url = {https://github.com/HassDhia/cardiosim}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Hass Dhia - Smart Technology Investments Research Institute
- Email: partners@smarttechinvest.com
- Web: [smarttechinvest.com/research](https://smarttechinvest.com/research)
