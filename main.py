import tweepy
from textblob import TextBlob
import pandas as pd
import datetime
import numpy as np
import time
import pickle
import requests
import os
from sklearn import tree 
import random as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation, metrics 
import matplotlib.pyplot as plt 
from matplotlib import style 

def get_counter(dataset):
    """
    Counts the number of occurences for each label
    """
    label_collection = [i[-1] for i in dataset]
    label_set = set(label_collection)
    counter = {}
    for label in label_set:
        counter[label] = 0
    for label in label_collection:
        counter[label] += 1
    return counter 

def get_keys(counter):
    """
    Returns keys from the counter dictionary
    """
    return(
        [key for key in counter]
    )

def get_values(counter):
    """
    Returns the counter values from the key value pair
    """
    return(
        [counter[key] for key in 
            get_keys(counter)
        ]
    )

def get_lowest_frequency(counter):
    """
    Returns the lowest value from get_values()
    """
    return min(
        get_values(counter)
    )

def get_datapoints_by_label(dataset, counter):
    """
    Returns all the datapoints with a given label
    """
    different_labels = get_keys(counter)
    datapoints_by_label = []
    for label in different_labels:
        temp_array = []
        for datapoint in dataset:            
            if float(datapoint[-1]) == float(label):                
                temp_array.append(datapoint)
        datapoints_by_label.append(temp_array)
    return datapoints_by_label

def rebalance(dataset, counter):
    """
    Obtains the label with the lowest frequency and creates a dataset
    that has this same amount of occurences for each label
    """
    # Get label with lowest frequency
    lowest_frequency = get_lowest_frequency(counter)
    print("\nObtaining datapoints so that there are always {} datapoints for each label\n".format(lowest_frequency))
    # Get list of datapoints by label
    datapoints_by_label = get_datapoints_by_label(dataset, counter)
    # Get only the number of datapoints per label as the lowest frequency
    rebalanced_data = []
    for d in datapoints_by_label:
        d = d[:lowest_frequency]
        rebalanced_data.extend(d)
    return rebalanced_data

class Tweet:
    """ 
    This class holds data about the tweet which will
    be used to obtain the features for the model
    """
    def __init__(self, text, length, polarity, subjectivity):
        self.text = text
        self.length = length
        self.polarity = polarity
        self.subjectivity = subjectivity

coins = [
    'ADA',
    'BTC',
    'DOGE',
    'ETH',
    'MIOTA',
]

# To store all the data about tweets
dataset = []

# Retrieve the features and labels of all tweets for each coin
for coin in coins:
    file_name = coin + ".txt"
    with open(file_name, 'r') as f:
        for line in f:
            item = line.split(",")
            f1, f2, f3, l = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            dataset.append([f1, f2, f3, l])

# Rebalance Dataset
dataset = rebalance(dataset, get_counter(dataset))

print("Dataset size: {}".format(len(dataset)))

""" GENETIC ALGORITHM """

# Probability of crossover between chromosomes
p_c = 1 # --> We always want there to be a crossover, so a probability of 1
# Probability of mutation
p_m = 0.2
# Population per generation
pop = 40
# Generations
gen = 50 

sub_chromosome_length = 13
chromosome_length = 2 * sub_chromosome_length

# Upper and lower bounds for decision tree hyperparameters, which are:
# --> min_weight_fraction_leaf
# --> min_impurity_split
lower_bound = 0
upper_bound = 1
# We need the precision to decode the chromosome value
precision = float(upper_bound-lower_bound) / float((2**sub_chromosome_length)-1)

# 13 for min_weight_fraction_leaf and 13 for min_impurity_split
XY0 = np.array([1,1,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0])

def decode(chromosome):
    """
    Convert the chromosome into an array
    of two elements, where the first one is the 
    min_weight_fraction leaf and the second one is
    min_impurity_split
    """
    start = 0
    end = sub_chromosome_length
    solution = []
    for i in range(2):
        item = chromosome[start:end]
        z = 0
        t = 1
        total_sum = 0
        for i in range(len(item)):
            number = item[-t]*2**z
            total_sum += number
            t = t + 1
            z = z + 1
        decoded = float((total_sum * precision) + lower_bound)
        solution.append(decoded)
        start = end
        end += sub_chromosome_length
    return solution


def create_classifier(solution):
    """
    Create an untrained decision tree with 
    the specified hyperparameters and return it
    """
    classifier = tree.DecisionTreeClassifier(min_weight_fraction_leaf=solution[0], min_impurity_split=solution[1])
    return classifier

def fitness_value(chromosome, clf, X_train, y_train, X_test, y_test):
    """
    Calculate the accuracy score and return it
    """
    # Get the weights and convert to np array
    solution = np.array(decode(chromosome))
    # Calculate accuracy
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    fitness_value = accuracy * 100
    return fitness_value

class Combination:

    """
    A class whose objects are used to store the combinations
    of min_weight_fraction_leaf and min_impurity_split
    created along the genetic algorithm process
    """
    
    def __init__(self, min_weight_fraction_leaf, min_impurity_split, accuracy):
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_split = min_impurity_split 
        self.accuracy = accuracy


def get_optimal_decision_tree(X_train, X_test, y_train, y_test):
    """
    The genetic algorithm process.
    Returns a trained classifier with the hyperparameters
    obtained from the genetic algorithm.
    """
    best_combinations = []

    """ Create a population """
    population = []
    for i in range(pop):
        chromosome = []
        for i in range(chromosome_length):
            # Generate random number
            random_number = rd.uniform(0,1)
            if random_number <= 0.5:
                chromosome.append(0)
            else:
                chromosome.append(1)
            solution = decode(chromosome)
            population.append(solution)

    """ Convert to np arrays """
    population = [np.array(i) for i in population]

    """ PROCESS """

    generation = 1
    # NOTE: temp make best_solution_in_each_generation a list
    # best_solution_in_each_generation = np.empty((0,chromosome_length))
    best_solution_in_each_generation = []

    print("STARTING GENETIC ALGORITHM CALCULATION")

    for i in range(gen):

        # For new population with new mutated children
        # NOTE: temp make parents normal list
        # new_population = np.empty((0,chromosome_length))
        new_population = []

        # Print generation to keep track
        print("Generation number {}".format(generation))

        # Keep track of family number
        family = 1

        for j in range(int(pop/2)):

            """ Keep track of family number """
            print("\tFamily number {}".format(family))

            """ Tournament selection """

            # Parents that will eventually have the children            
            parents = []

            # We want two parents
            for i in range(2):

                battle_troops = []

                # Pick 3 random warriors
                warrior_1_index = np.random.randint(0,len(population))
                warrior_2_index = np.random.randint(0,len(population))
                warrior_3_index = np.random.randint(0,len(population))

                # Make sure they are not the same
                while warrior_1_index == warrior_2_index:
                    warrior_1_index = np.random.randint(0,len(population))
                while warrior_2_index == warrior_3_index:
                    warrior_2_index = np.random.randint(0,len(population))
                while warrior_1_index == warrior_3_index:
                    warrior_3_index = np.random.randint(0,len(population))

                # Get the actual items
                warrior_1 = population[warrior_1_index]
                warrior_2 = population[warrior_2_index]
                warrior_3 = population[warrior_3_index]

                """ Calculate the fitness value of each """
                w1_fitness_value = fitness_value(warrior_1, create_classifier(decode(warrior_1)), X_train, y_train, X_test, y_test)
                w2_fitness_value = fitness_value(warrior_2, create_classifier(decode(warrior_2)), X_train, y_train, X_test, y_test)
                w3_fitness_value = fitness_value(warrior_3, create_classifier(decode(warrior_3)), X_train, y_train, X_test, y_test)

                # The one with the highest difference is the winner
                winner = warrior_1
                if w1_fitness_value == max(w1_fitness_value,w2_fitness_value,w3_fitness_value):
                    winner = warrior_1
                elif w2_fitness_value == max(w1_fitness_value,w2_fitness_value,w3_fitness_value):
                    winner = warrior_2
                else:
                    winner = warrior_3

                # Append to parents
                parents.append(winner)

            # Get parents
            parent_1 = parents[0]
            parent_2 = parents[1]

            # Create empty arrays for the children
            child_1 = np.empty((0,chromosome_length))
            child_2 = np.empty((0,chromosome_length))

            mutated_child_1_decoded_sum = 0
            mutated_child_2_decoded_sum = 0

            # Generate random number for probability of crossover
            probability_crossover = np.random.rand()

            # Determine if to do a crossover
            if probability_crossover < p_c:

                # Create two random cutoff points
                cutoff_point_1 = np.random.randint(0,chromosome_length)
                cutoff_point_2 = np.random.randint(0,chromosome_length)

                # Make sure the two cutoff points are not the same
                while cutoff_point_1 == cutoff_point_2:
                    cutoff_point_2 = np.random.randint(0,chromosome_length)

                """ Create segments """
                if cutoff_point_1 < cutoff_point_2:

                    # Middle
                    middle_segment_1 = parent_1[cutoff_point_1:cutoff_point_2+1]
                    middle_segment_2 = parent_2[cutoff_point_1:cutoff_point_2+1]
                    # --> Must include +1 since the second range value is not included

                    # Parent 1
                    first_segment_1 = parent_1[:cutoff_point_1]
                    second_segment_1 = parent_1[cutoff_point_2+1:]

                    # Parent 2
                    first_segment_2 = parent_2[:cutoff_point_1]
                    second_segment_2 = parent_2[cutoff_point_2+1:]

                    # Create new chromosome
                    child_1 = np.concatenate((first_segment_1,middle_segment_2,second_segment_1))
                    child_2 = np.concatenate((first_segment_2,middle_segment_1,second_segment_2))

                else:

                    # Middle
                    middle_segment_1 = parent_1[cutoff_point_2:cutoff_point_1+1]
                    middle_segment_2 = parent_2[cutoff_point_2:cutoff_point_1+1]
                    # --> Must include +1 since the second range value is not included

                    # Parent 1
                    first_segment_1 = parent_1[:cutoff_point_2]
                    second_segment_1 = parent_1[cutoff_point_1+1:]

                    # Parent 2
                    first_segment_2 = parent_2[:cutoff_point_2]
                    second_segment_2 = parent_2[cutoff_point_1+1:]

                    # Create new chromosome
                    child_1 = np.concatenate((first_segment_1,middle_segment_2,second_segment_1))
                    child_2 = np.concatenate((first_segment_2,middle_segment_1,second_segment_2))

            else:

                # If the probability for crossover is not lower than the random number
                child_1 = parent_1
                child_2 = parent_2

            """ Mutations """

            # ** Mutating Child 1 **
            # This is for going through each element of the child
            mutated_child_1 = []
            t = 0
            for i in child_1:
                # Generate random number to compare with probability of mutation
                random_number = np.random.rand()
                # If random number smaller than probability of mutation
                if random_number < p_m:
                    if child_1[t] == 0:
                        child_1[t] = 1
                    else:
                        child_1[t] = 0
                    t = t + 1
                    mutated_child_1 = child_1
                else:
                    mutated_child_1 = child_1

            # ** Mutating Child 2 **
            # This is for going through each element of the child
            mutated_child_2 = []
            t = 0
            for i in child_2:
                # Generate random number to compare with probability of mutation
                random_number = np.random.rand()
                # If random number smaller than probability of mutation
                if random_number < p_m:
                    if child_2[t] == 0:
                        child_2[t] = 1
                    else:
                        child_2[t] = 0
                    t = t + 1
                    mutated_child_2 = child_2
                else:
                    mutated_child_2 = child_2

            """ Get fitness values """
            mutated_child_1_fitness_value = fitness_value(mutated_child_1, create_classifier(decode(mutated_child_1)), X_train, y_train, X_test, y_test)
            mutated_child_2_fitness_value = fitness_value(mutated_child_2, create_classifier(decode(mutated_child_2)), X_train, y_train, X_test, y_test)

            """ Add the children to the new population """
            new_population.append(mutated_child_1)
            new_population.append(mutated_child_2)

            """ Keep track of family number """
            family += 1

        """ Find the best and worst in the generation """

        # Set children and fitness values just to start off with
        best_in_generation = new_population[0]
        best_fitness_value_in_generation = float(0)
        worst_in_generation = [1]
        worst_fitness_value_in_generation = float(0)

        fitness_values = []

        # Get all fitness values
        for i in range(len(new_population)):
            fitness_values.append(
                fitness_value(new_population[i], create_classifier(decode(new_population[i])), X_train, y_train, X_test, y_test)
            )

        # Highest
        highest_index = 0
        highest = max(fitness_values)
        for i in range(len(new_population)):
            if fitness_value(new_population[i], create_classifier(decode(new_population[i])), X_train, y_train, X_test, y_test) == highest:
                best_in_generation = new_population[i]
                highest_index = i

        # Lowest
        lowest_index = 0
        lowest = min(fitness_values)
        for i in range(len(new_population)):
            if fitness_value(new_population[i], create_classifier(decode(new_population[i])), X_train, y_train, X_test, y_test) == lowest:
                worst_in_generation = new_population[i]
                lowest_index = i

        """ Save the best result in each generation """
        best_solution_in_each_generation.append(best_in_generation)

        """ Replace the worst with the best """
        new_population[lowest_index] = new_population[highest_index]

        """ Make the new population the actual population """
        population = new_population

        generation += 1

    all_best_solutions = []
    for s in best_solution_in_each_generation:
        all_best_solutions.append(
            fitness_value(s, create_classifier(decode(s)), X_train, y_train, X_test, y_test)
        )

    """ Get the best solution """
    highest_fitness_value = max(all_best_solutions)
    best_solution = best_solution_in_each_generation[0]
    for s in best_solution_in_each_generation:
        if fitness_value(s, create_classifier(decode(s)), X_train, y_train, X_test, y_test) == highest_fitness_value:
            best_solution = s

    classifier = create_classifier(decode(best_solution)).fit(X_train, y_train)

    """ Plot the graph """
    x = [(i+1) for i in range(gen)]
    y = all_best_solutions
    plt.plot(x,y)
    plt.axhline(y=highest_fitness_value, color='r', linestyle='-')
    plt.title("Optimal hyperparameters", fontsize=20, fontweight='bold')
    plt.xlabel("Generation", fontsize=15, fontweight='bold')
    plt.ylabel("Best Fitness Value", fontsize=15, fontweight='bold')
    plt.show()

    return classifier


def get_best_model(dataset):
    """
    Creates the dataset and trains the model, 
    optimises it with the genetic algorithm 
    and returns the model and its accuracy
    """    
    best_model = 0
    # Shuffle Dataset
    rd.shuffle(dataset)   
    # Create dataset
    features = []
    for i in dataset:
        i = i[:-1]
        features.append(i)                        
    labels = []
    for i in dataset:
        labels.append(i[-1])
    # Transform
    scaler = MinMaxScaler()
    features = list(scaler.fit_transform(features))
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)                    
    best_model = get_optimal_decision_tree(X_train, X_test, y_train, y_test) 
    accuracy = metrics.accuracy_score(
        y_test,
        best_model.predict(X_test)
    )
    return best_model, accuracy 

best_model, accuracy = get_best_model(dataset)

print("Accuracy = {}%".format(round(accuracy*100,2)))