import random
import numpy as np
import matplotlib.pyplot as plt

# Cities coordinates
cities_coords = {
    'A': (2,5),
    'B': (1,5),
    'C': (2,3),
    'D': (3,4),
    'E': (2,4),
    'F': (3,3),
    'G': (5,5),
    'H': (1,1),
    'I': (1,4),
    'J': (1,2),
    'K': (2,2),
    'L': (4,4),
    'M': (2,1),
    'N': (3,1)
}

# Parameters
num_ants = 25
num_iterations = 10

# Hyper parameters
alpha = 0.7
beta = 5.0
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
    city_names = list(cities_coords.keys())
    city_coords = list(cities_coords.values())
    num_cities = len(cities_coords)

    distance_matrix = calculate_distance_matrix(cities_coords)
    heuristic_info = 1 / (distance_matrix + 1e-10)
    pheromone_matrix = np.ones((num_cities, num_cities))

    best_path = None
    best_path_distance = float('inf')

    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 10))

    for iteration in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            current_city = random.randint(0, num_cities-1)
            path = [current_city]
            unvisited_cities = set(range(num_cities))
            unvisited_cities.remove(current_city)

            while unvisited_cities:
                probabilities = []
                for next_city in unvisited_cities:
                    pheromone = pheromone_matrix[current_city][next_city] ** alpha
                    heuristic = heuristic_info[current_city][next_city] ** beta
                    probabilities.append(pheromone * heuristic)

                probabilities_sum = sum(probabilities)
                probabilities = [p / probabilities_sum for p in probabilities]

                # Randomly select the next city
                next_city = random.choices(list(unvisited_cities), weights=probabilities, k=1)[0]
                path.append(next_city)
                unvisited_cities.remove(next_city)
                current_city = next_city

            all_paths.append(path)

        # Evaporate pheromone
        pheromone_matrix *= (1 - evaporation_rate)

        # Select best path and increase pheromone concentration
        for path in all_paths:
            path_distance = calculate_path_distance(path,distance_matrix)
            if path_distance < best_path_distance:
                best_path = path
                best_path_distance = path_distance

            pheromone_to_deposit = pheromone_deposit / path_distance
            for k in range(num_cities - 1):
                pheromone_matrix[path[k]][path[k+1]] += pheromone_to_deposit
            pheromone_matrix[path[-1]][path[0]] += pheromone_to_deposit

        ax.clear()

        # plot the paths using pheromone
        max_pheromone = np.max(pheromone_matrix)
        for row in range(num_cities):
            for col in range(row, num_cities):
                if pheromone_matrix[row][col] > 1e-9:
                    ax.plot([city_coords[row][0], city_coords[col][0]],
                            [city_coords[row][1], city_coords[col][1]],
                            color='red', linestyle=':',
                            linewidth = 2 * (pheromone_matrix[row][col] / max_pheromone),
                            alpha=0.6)
                    
        if best_path:
            best_path_coords = [city_coords[j] for j in best_path]
            best_path_coords.append(best_path_coords[0])
            x_coords, y_coords = zip(*best_path_coords)
            ax.plot(x_coords, y_coords, color='blue', linewidth=2, label='Best Path')

        ax.scatter([c[0] for c in city_coords], [c[1] for c in city_coords], color='green', s=100, zorder=3)
        for name, (x, y) in cities_coords.items():
            ax.text(x + 5, y + 5, name, fontsize=12)
        
        ax.set_title(f"Iteration {iteration + 1}, Best Distance: {best_path_distance:.2f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        plt.draw()
        plt.pause(0.5)

    plt.ioff()
    print("ACO algorithm completed.")
    best_path_cities = [city_names[j] for j in best_path]
    print(f"Best path: {' -> '.join(best_path_cities)} -> {best_path_cities[0]}")
    print(f"Best path distance: {best_path_distance:.2f}")

    ax.set_title(f"Final Best Path by Distance: {best_path_distance:.2f}")
    plt.show()

if __name__ == '__main__':
    execute_main()
    