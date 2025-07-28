import matplotlib.pyplot as plt
import argparse
import json

from evolution.genetic_algorithm import GeneticAlgorithm

def main():

    parser = argparse.ArgumentParser(prog="Neuroevolution BipedalWalker-v3")

    parser.add_argument("-r", "--render", action="store_true", help="Renders a portion of genomes gifs and stores them in /assets/gifs/.")

    arg = parser.parse_args()

    input_size = 24
    hidden1_size = 32
    hidden2_size = 32
    output_size = 4

    population_size = 50
    mutation_rate = 0.1
    mutation_strength = 0.3
    
    crossover_rate = 0.75
    tournament_size = 7
    elitism = 4
    generations = 20000

    render = True if arg.render else False
    
    ga = GeneticAlgorithm(input_size, hidden1_size, hidden2_size, output_size, population_size, mutation_rate, mutation_strength, tournament_size, crossover_rate, elitism, render)

    best_fitness_history = []
    avg_fitness_history = []
    worst_fitness_history = []

    for gen in range(generations):
        population = ga.evaluate_population(gen)
        fitness_values = [fitness for _, fitness in population]
        best_fitness = fitness_values[0]
        worst_fitness = fitness_values[-1]
        avg_fitness = sum(fitness_values) / len(fitness_values)

        print(f"Generation {gen + 1} | Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Worst: {worst_fitness:.2f}")

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        worst_fitness_history.append(worst_fitness)

        ga.next_generation(population)

    with open("assets/data/fitness_stats.json", "w") as f:
        json.dump({
            "best": best_fitness_history,
            "avg": avg_fitness_history,
            "worst": worst_fitness_history
        }, f)

    plt.plot(best_fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.plot(worst_fitness_history, label="Worst Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(bbox_to_anchor=(0.99, 1.12), ncol=3)

    plt.savefig("assets/figures/fitness_plot.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
