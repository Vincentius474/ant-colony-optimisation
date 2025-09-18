import random
import numpy as np
import matplotlib.pyplot as plt

# Cities coordinates
cities_coords = {
    'A': (2,5),
    'B': (2,5),
    'C': (2,5),
    'D': (2,5),
    'E': (2,5),
    'F': (2,5),
    'G': (2,5),
    'H': (2,5),
    'I': (2,5),
    'J': (2,5),
    'K': (2,5),
    'L': (2,5),
    'M': (2,5),
    'N': (2,5)
}

# Parameters
num_ants = 20
num_iterations = 100

# Hyper parameters
alpha = 0.6
beta = 0.7
evaporation_rate = 0.5
pheromone_deposit = 100.0

# Construct the distance metrix
def calculate_distance_matrix(cities_coords):
    num_cities = len(cities_coords)
    distance_matrix = np.zeros((num_cities, num_cities))
    list_values = list(cities_coords.values())
    for i in range(num_cities):
        for j in range(i, num_cities):
            distance = np.linalg.norm(np.array(list_values[i]) - np.array(list_values[j]))
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    return distance_matrix

# Calculation of path distance
def calculate_path_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += distance_matrix[path[i]][path[i+1]]
    total_distance += distance_matrix[path[-1]][path[0]]
    return total_distance

# Execution of basic ant colony optimization
def execute_main():
    list_keys = list(cities_coords.keys())
    list_values = list(cities_coords.values())
    num_cities = len(cities_coords)

    distance_matrix = calculate_distance_matrix(cities_coords)
    heuristic_info = 1 / (distance_matrix * 1e-10)

    pheromone_matrix = np.ones((num_cities, num_cities))
    best_path = None
    best_path_distance = float('-inf')

    plt.ion
    fig , ax = plt.subplots(figsize=(10,10))

    print('Starting the Ant Colony Optimisation')
    
    for i in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            current_city = np.random.randint(0, num_cities - 1)
            path = [current_city]
            unvisited_cities = set(range(cities_coords))
            unvisited_cities.remove(current_city)

            while unvisited_cities:
                probabilities = []
                for next_city in unvisited_cities:
                    pheromone = pheromone_matrix[current_city][next_city] * alpha
                    heuristic = heuristic_info[current_city][next_city] * best_path
                    probabilities.append(pheromone * heuristic)

                probabilities_sum = sum(probabilities)
                probabilities = [p/probabilities_sum for p in probabilities]

                next_city = np.random.choice(list(unvisited_cities), weights=probabilities, k=1)[0]
                path.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            all_paths.append(path)

        pheromone_matrix *= (1 - evaporation_rate)
        for path in all_paths:
            path_distance = calculate_path_distance(path, distance_matrix)
            if path_distance < best_path_distance:
                best_path_distance = path_distance
                best_path = path

            pheromone_to_deposit = pheromone_deposit / path_distance
            for k in range(num_cities - 1):
                pheromone_matrix[path[k]][path[k + 1]] += pheromone_to_deposit
            pheromone_matrix[path[-1]][path[0]] += pheromone_to_deposit

        ax.clear()

        max_pheromone = np.max(pheromone_matrix)
        for row in range(num_cities):
            for col in range(row, num_cities):
                if pheromone_matrix[row][col] > 1e-9:
                    ax.plot([cities_coords[row][0], cities_coords[col][0]],
                            [cities_coords[row][1], cities_coords[col][1]],
                            color='blue',
                            linestyle='-',
                            linewidth=2 * (pheromone_matrix[row][col] / max_pheromone),
                            alpha=0.7)
        
        if best_path:
            best_path_coords = [cities_coords[j] for j in best_path]
            best_path_coords.append(best_path_coords[0])
            x_coords, y_coords = zip(*best_path_coords)
            ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Best Path')

        ax.scatter([c[0] for c in cities_coords], [c[1] for c in cities_coords], color='green', s=100, zorder=3)
        for name, (x, y) in cities_coords.items():
            ax.text(x + 5, y + 5, name, fontsize=12)
        
        ax.set_title(f"Iteration {i + 1}, Best Distance: {best_path_distance:.2f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        plt.draw()
        plt.pause(0.5)

    plt.ioff()
    print("ACO algorithm completed.")
    best_path_cities = [list_keys[j] for j in best_path]
    print(f"Best path: {' -> '.join(best_path_cities)} -> {best_path_cities[0]}")
    print(f"Best path distance: {best_path_distance:.2f}")

    ax.set_title(f"Final Best Path by Distance: {best_path_distance:.2f}")
    plt.show()

if __name__ == '__main__':
    execute_main()
    