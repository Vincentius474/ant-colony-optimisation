# Ant Colony Optimization Simulations

## Overview

This repository contains two Python implementations of **Ant Colony Optimization (ACO)** algorithms, showcasing how artificial ants can collaboratively solve optimization and pathfinding problems using pheromone-based heuristics.

---

## Files

| `basic_ant_colony_optimisation.py` -> Basic ACO implementation for solving the Travelling Salesman Problem (TSP).   
| `Advanced_ACO_pathfinding.py` -> Advanced ACO-based pathfinding simulation on a 2D grid with obstacles, diffusion, and interactive visualization.

---

## 1. Basic Ant Colony Optimization (`basic_ant_colony_optimisation.py`)

### Description

This script demonstrates a **basic ACO algorithm** to find the shortest path visiting all cities exactly once and returning to the start (TSP). It uses:

* Probabilistic path selection based on pheromone trails and distance heuristics.
* Dynamic pheromone evaporation and reinforcement.
* Real-time visualization of paths and pheromone strengths.

### Key Features

* Solves a TSP with predefined city coordinates.
* Visualizes pheromone trails strengthening over iterations.
* Highlights the current best path in blue.

### Parameters

| Parameter           | Description                            | Default |
| ------------------- | -------------------------------------- | ------- |
| `num_ants`          | Number of simulated ants               | 25      |
| `num_iterations`    | Number of iterations                   | 10      |
| `alpha`             | Pheromone importance                   | 0.7     |
| `beta`              | Distance heuristic importance          | 5.0     |
| `evaporation_rate`  | Pheromone decay rate                   | 0.5     |
| `pheromone_deposit` | Amount of pheromone deposited per path | 100.0   |

### Run

```bash
python basic_ant_colony_optimisation.py
```

### Visualization

* Green dots represent cities.
* Red dotted lines indicate pheromone-weighted edges.
* Blue solid line shows the current best tour.

---

## 2. Advanced ACO Pathfinding (`Advanced_ACO_pathfinding.py`)

### Description

This program extends ACO to **2D pathfinding**, simulating ants navigating from a nest to food sources while avoiding obstacles.
It incorporates **pheromone diffusion**, **evaporation**, and **reinforcement**, allowing ants to collectively discover efficient routes.

### Key Features

* Grid-based environment with obstacles, nest, and food sources.
* Configurable via interactive **prompt mode** or automatic random generation.
* Real-time heatmap visualization of pheromone trails.
* Adaptive exploration and path reinforcement dynamics.

### Parameters

| Parameter            | Description                                   | Default |
| -------------------- | --------------------------------------------- | ------- |
| `GRID_SIZE`          | Size of the simulation grid                   | 50      |
| `NUM_ANTS`           | Number of ants                                | 40      |
| `NUM_OBSTACLES`      | Number of obstacles                           | 5       |
| `NUM_SOURCES`        | Number of food sources                        | 3       |
| `NUM_ITERATIONS`     | Simulation steps                              | 500     |
| `EVAPORATION_RATE`   | Pheromone decay per iteration                 | 0.5     |
| `PHEROMONE_DEPOSIT`  | Pheromone added per path                      | 100.0   |
| `EXPLORATION_FACTOR` | Probability of random exploration             | 0.3     |
| `DIFFUSION_RATE`     | Pheromone diffusion between neighboring cells | 0.1     |

### Run

```bash
python Advanced_ACO_pathfinding.py
```

**Optional:** Run interactively to manually set parameters and environment.

```bash
python Advanced_ACO_pathfinding.py
```

Then choose:

```
Run in prompt mode? (Y/N): y
```

### Visualization

* Grey = Nest
* Brown = Obstacles
* Green = Food sources
* Red = Ants carrying food
* Lime = Ants searching
* Yellow-to-red overlay = Pheromone concentration

---

## Dependencies

Both scripts require:

```bash
pip install numpy matplotlib
```

---

## License

This project is open source under the **MIT License**.

