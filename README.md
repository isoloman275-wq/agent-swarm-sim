# agent-swarm-sim

A from-scratch **evolutionary simulation** of competing agents — a survival-of-the-fittest
swarm. Prey, predators and a disease layer compete on a fluctuating world. Fitness is
selection pressure; survivors reproduce with mutation; the weak are culled.

Pure Python (standard library only). No external dependencies.

## Run
```bash
python3 agent_swarm_evolution.py --live --seed 42 --pred 8 --food 14 --delay 0.15
```
Flags: `--seed`, `--gens`, `--pred` (predators), `--food` (food patches), `--live` (terminal
animation), `--csv` (export run to CSV), `--frames` (export text frames for video/GIF).

## What it demonstrates
- Agent-based modelling & selection
- Genetic drift via mutation on inherited "genes" (speed, sense, metabolism)
- Emergent predator/prey oscillation and disease outbreaks
- Clean, dependency-free Python suitable for teaching or extension

Built by NZ1Labs.
