import numpy as np
import random

from evolution.genome import create_genome
from environment.bipedalwalker_runner import evaluate_genome

class GeneticAlgorithm:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, population_size, mutation_rate, mutation_strength, tournament_size, crossover_rate, elitism, render):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.genome_length = self.input_size * self.hidden1_size + self.hidden1_size + self.hidden1_size * self.hidden2_size + self.hidden2_size + self.hidden2_size * self.output_size + self.output_size
        self.population = self.initialize_population()
        self.render = render

    def initialize_population(self):
        return [create_genome(self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)
                for _ in range(self.population_size)]

    def mutate(self, genome):
        mask = np.random.random(self.genome_length) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_strength, self.genome_length)

        return genome + mask * noise

    def crossover(self, parent1, parent2):
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        mask = np.random.random(self.genome_length) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2

    def tournament_selection(self, fitness_scores):
        tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        tournament_individuals = [fitness_scores[i] for i in tournament_indices]
        tournament_fitnesses = [individual[1] for individual in tournament_individuals]
        winner_index = np.argmax(tournament_fitnesses)

        return tournament_individuals[winner_index][0]

    def evaluate_population(self, generation):
        fitness_scores = []
        seed = random.randint(0, 1000)
        for i, genome in enumerate(self.population):
            fitness = evaluate_genome(genome, self.input_size, self.hidden1_size, self.hidden2_size, self.output_size, i, generation, seed, self.render)
            fitness_scores.append((genome, fitness))
        

        fitness_scores.sort(key=lambda x:x[1], reverse=True)
        return fitness_scores

    def next_generation(self, fitness_scores):
        new_population = []
        
        if self.elitism < 0:
            for i in range(elitism):
                new_population.append(fitness_scores[i][0])

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population

