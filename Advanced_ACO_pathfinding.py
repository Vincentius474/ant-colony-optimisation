import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

GRID_SIZE = 50
NUM_ANTS = 40
NUM_OBSTACLES = 5
NUM_SOURCES = 3
NUM_ITERATIONS = 500

EVAPORATION_RATE = 0.5
PHEROMONE_DEPOSIT = 100.0
PHEROMONE_MIN = 0.01
EXPLORATION_FACTOR = 0.3
DIFFUSION_RATE = 0.1

EMPTY_CELL, NEST, OBSTACLE, FOOD_SOURCE = 0, 1, 2, 3

DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, 1), (1, -1)
]

class Ant:

    def __init__(self, nest_coords):
        self.x, self.y = nest_coords
        self.laden = False
        self.path = []
        self.target_food = None

    def _distance(self, coords_a, coords_b):
        return sum(abs(i - j) for i, j in zip(coords_a, coords_b))

    def select_move(self, grid, pheromones, nest):
        target = nest if self.laden else self.target_food
        possible_moves = []

        for dx, dy in DIRECTIONS:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny] != OBSTACLE:
                pheromone = pheromones[nx][ny] + 1e-5
                distance_factor = 1 / (self._distance((nx, ny), target) + 1) if target else 1
                score = (pheromone ** 2) * distance_factor
                if random.random() < EXPLORATION_FACTOR:
                    score *= random.uniform(0.5, 1.5)
                possible_moves.append(((nx, ny), score))
        
        if not possible_moves:
            return None
        
        total = sum(score for _, score in possible_moves)
        probs = [score / total for _, score in possible_moves]
        return random.choices([pos for pos, _ in possible_moves], weights=probs)[0]
    
    def move(self, grid, pheromones, food_sources, nest):
        if not self.laden and not self.target_food and food_sources:
            self.target_food = random.choice(food_sources)

        next_pos = self.select_move(grid, pheromones, nest)
        dist_a = self._distance((self.x, self.y), self.target_food)
        if next_pos:
            self.x, self.y = next_pos
            self.path.append(next_pos)

    def update(self, grid, nest):
        if not self.laden and grid[self.x, self.y] == FOOD_SOURCE:
            self.laden = True
            # grid[self.x, self.y] = EMPTY_CELL
            self.path = [(self.x, self.y)]
        elif self.laden and (self.x, self.y) == nest:
            self.laden = False
            self.target_food = None
            return self.path
        return None

def create_environment(prompt_mode=False):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    if prompt_mode:
        print(f"\nGrid size: {GRID_SIZE} x {GRID_SIZE}")
        print(f"All values for x & y coordinates should be in the range [0 - {GRID_SIZE-1}]\n")
        x_nest = int(input(f"Nest x (0-{GRID_SIZE - 1}): "))
        y_nest = int(input(f"Nest y (0-{GRID_SIZE - 1}): "))
        grid[x_nest][y_nest] = NEST

        num_sources = int(input("\nNumber of food sources: ") or NUM_SOURCES)
        for i in range(num_sources):
            print(f"\nFood source {i + 1}:")
            x = int(input("x: "))
            y = int(input("y: "))
            grid[x][y] = FOOD_SOURCE 

        num_obstacles = int(input("\nNumber of obstacles: ") or NUM_OBSTACLES)
        for i in range(num_obstacles):
            print(f"\nObstacle {i + 1}:")
            x = int(input("x: "))
            y = int(input("y: "))
            grid[x][y] = OBSTACLE

    else:

        x_nest, y_nest = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        grid[x_nest][y_nest] = NEST

        for _ in range(NUM_SOURCES):
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            while grid[x][y] != EMPTY_CELL:
                x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            grid[x][y] = FOOD_SOURCE

        for _ in range(NUM_OBSTACLES):
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if grid[x][y] == EMPTY_CELL:
                grid[x][y] = OBSTACLE

    return grid

def evaporate_pheromones(pheromones):
    pheromones *= (1 - EVAPORATION_RATE)
    pheromones[pheromones < PHEROMONE_MIN] = 0.0

def diffuse_pheromones(pheromones):
    new_pheromone = pheromones.copy()
    for x in range(1, GRID_SIZE - 1):
        for y in range(1, GRID_SIZE - 1):
            if pheromones[x, y] > 0:
                total = sum(
                    pheromones[x + dx, y + dy]
                    for dx, dy in DIRECTIONS
                    if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE
                )
                avg = total / len(DIRECTIONS)
                new_pheromone[x, y] += avg * DIFFUSION_RATE
    return new_pheromone

def reinforce_pheromones(pheromones, path):
    if not path:
        return
    for (x, y) in path:
        pheromones[x][y] += PHEROMONE_DEPOSIT / (len(path) ** 0.5)
    if len(path) > 1:
        pheromones[path[0][0]][path[0][1]] += PHEROMONE_DEPOSIT * 1.5
        pheromones[path[-1][0]][path[-1][1]] += PHEROMONE_DEPOSIT * 1.5

def display_environment(grid, pheromones, ants, iteration, best_path=None):

    plt.clf()
    cmap = mcolors.ListedColormap(['white', 'grey', 'brown', 'green'])
    plt.imshow(grid, cmap=cmap)

    for ant in ants:
        plt.plot(ant.y, ant.x, '^', color='red' if ant.laden else 'lime', markersize=4)

    if np.max(pheromones) > 0:
        plt.imshow(pheromones / np.max(pheromones), cmap='hot', alpha=0.19)

    if best_path:
        path_x, path_y = zip(*best_path)
        plt.plot(path_y, path_x, 'g-', linewidth=2)
    
    plt.title(f'Ant Colony Optimization\nIteration {iteration + 1} of {NUM_ITERATIONS}')
    plt.pause(0.01)


def main(prompt_mode=False):
    global EVAPORATION_RATE, PHEROMONE_DEPOSIT, EXPLORATION_FACTOR, DIFFUSION_RATE

    if prompt_mode:
        print("\n Hyperparameter Configuration \n")
        EVAPORATION_RATE = float(input("EVAPORATION_RATE (0-1): ") or EVAPORATION_RATE)
        PHEROMONE_DEPOSIT = float(input("PHEROMONE_DEPOSIT: ") or PHEROMONE_DEPOSIT)
        EXPLORATION_FACTOR = float(input("EXPLORATION_FACTOR (0-1): ") or EXPLORATION_FACTOR)
        DIFFUSION_RATE = float(input("DIFFUSION_RATE (0-0.5): ") or DIFFUSION_RATE)

    grid = create_environment(prompt_mode)
    food_sources = [tuple(pos) for pos in np.argwhere(grid == FOOD_SOURCE)]
    nest = tuple(np.argwhere(grid == NEST)[0])
    pheromones = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    ants = [Ant(nest) for _ in range(NUM_ANTS)]

    best_path = None
    best_score = float('-inf')
    path_counts = defaultdict(int)

    plt.figure(figsize=(10, 10))

    for iteration in range(NUM_ITERATIONS):
        for ant in ants:
            ant.move(grid, pheromones, food_sources, nest)
            path = ant.update(grid, nest)

            if path:
                reinforce_pheromones(pheromones, path)
                key = tuple(path)
                path_counts[key] +=1
                score = len(path) / (path_counts[key] ** 0.5)
                if score > best_score:
                    best_score, best_path = score, path

        evaporate_pheromones(pheromones)
        pheromones = diffuse_pheromones(pheromones)

        if iteration % 5 == 0:
            display_environment(grid, pheromones, ants, iteration, best_path)

    display_environment(grid, pheromones, ants, iteration, best_path)

    if best_path:
        print(f"\nBest path found (Length = {len(best_path)})")
        print("Path coordinates:")
        coords = []
        for step, (x, y) in enumerate(best_path, 1):
            coords.append(f"[{x},{y}] ->")
        print(coords)
    else:
        print("No complete path found")

    plt.show()

if __name__ == "__main__":
    user_choice = input("Run in prompt mode? (Y/N): ").strip().lower()
    main(prompt_mode=(user_choice == 'y'))