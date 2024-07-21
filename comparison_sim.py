import numpy as np
import networkx as nx
import random

# sample subject area from distribution
def sample_subjects():
    dist = [('Analysis', 0.15), ('Algebra', 0.15), ('Combinatorics, Statistics, and Probability', 0.25), 
            ('Applied Math', 0.2), ('Geometry and Topology', 0.25)]
    ss = [1, 2, 3]
    p = [0.1, 0.8, 0.1]
    outcomes, probabilities = zip(*dist)
    sample_size = np.random.choice(ss, p=p)
    samples = np.random.choice(outcomes, size=sample_size, replace=False, p=probabilities)
    return samples.tolist()

# sample gender from distribution 
def sample_gender(m):
    student_distribution = [('Woman', 0.52), ('Man', 0.48)]
    mentor_distribution = [('Woman', 0.46), ('Man', 0.54)]
    if m == True:
        dist = mentor_distribution
        ss = [1]
        p = [1]
    else:
        dist = student_distribution
        ss = [1, 2]
        p = [0.5, 0.5]
    outcomes, probabilities = zip(*dist)
    sample_size = np.random.choice(ss, p=p)
    samples = np.random.choice(outcomes, size=sample_size, replace=False, p=probabilities)
    return samples.tolist()

# sample race from distribution
def sample_race(m):
    ssf = 100 / 79
    msf = 100 / 60
    student_distribution = [('Asian', 0.27*ssf), ('White', 0.24*ssf), ('Hispanic or Latino', 0.18*ssf), 
                            ('Black or African American', 0.08*ssf), ('American Indian or Alaska Native', 
                            0.01*ssf), ('Native Hawaiian or other Pacific Islander', 0.01*ssf)]
    mentor_distribution = [('Asian', 0.18*msf), ('White', 0.27*msf), ('Hispanic or Latino', 0.09*msf), 
                           ('Black or African American', 0.04*msf), ('American Indian or Alaska Native', 
                            0.01*msf), ('Native Hawaiian or other Pacific Islander', 0.01*msf)]
    if m == True:
        dist = mentor_distribution
        ss = [1, 2, 3]
        p = [0.8, 0.16, 0.04]
    else:
        dist = student_distribution
        ss = [1, 2, 3]
        p = [0.8, 0.16, 0.04]
    outcomes, probabilities = zip(*dist)
    sample_size = np.random.choice(ss, p=p)
    samples = np.random.choice(outcomes, size=sample_size, replace=False, p=probabilities)
    return samples.tolist()

# sample language from distribution
def sample_language():
    dist = [('Spanish', 0.926), ('Chinese', 0.074)]
    ss = [0, 1, 2]
    p = [0.844, 0.12, 0.036]
    outcomes, probabilities = zip(*dist)
    sample_size = np.random.choice(ss, p=p)
    samples = np.random.choice(outcomes, size=sample_size, replace=False, p=probabilities)
    return samples.tolist()

# sample preference weight vector from distribution
def sample_preference_weights():
    subj_weight = random.uniform(0, 100)
    gender_weight = random.uniform(0, 40)
    race_weight = random.uniform(0, 40)
    language_weight = random.uniform(0, 20)
    norm_sum = subj_weight + gender_weight + race_weight + language_weight
    pre_norm = [subj_weight, gender_weight, race_weight, language_weight]
    return [x * 100 / norm_sum for x in pre_norm]

# call on above sampling functions to generate applicant preferences and mentor profiles
# store as dicts of tuples, along with a nested dict for computed edge weights between each applicant 
# and mentor
def generate_dataset(num_students, num_mentors):
    weights_dict = {}
    student_prefs_dict = {}
    students = {}
    mentors = {}

    for i in range(num_students):
        subjects = sample_subjects()
        gender = sample_gender(False)
        race = sample_race(False)
        language = sample_language()
        students[f'X{i+1}'] = [subjects, gender, race, language]
        student_prefs_dict[f'X{i+1}'] = sample_preference_weights()

    for i in range(num_mentors):
        subjects = sample_subjects()
        gender = sample_gender(True)
        race = sample_race(True)
        language = sample_language()
        mentors[f'Y{i+1}'] = [subjects, gender, race, language]

    for student in students.keys():
        weights_dict[student] = {}
        for mentor in mentors.keys():
            intersects = [1 if not set(students[student][i]).isdisjoint(set(mentors[mentor][i])) 
                        else 0 for i in range(4)]
            weights_dict[student][mentor] = (sum([student_prefs_dict[student][i] if intersects[i] == 1 
                        else 0 for i in range(4)]), sum(intersects))
    
    return students, mentors, weights_dict

# create graph from two bipartite classes
def create_bipartite_graph(students, mentors, weights_dict):
    X = [f'X{i+1}' for i in range(len(students))]
    Y = [f'Y{i+1}' for i in range(len(mentors))]
    
    B = nx.Graph()
    B.add_nodes_from(X, bipartite=0)
    B.add_nodes_from(Y, bipartite=1)
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            B.add_edge(x, y, weight=weights_dict[x][y][0])
    
    return B, X, Y

# compute and return max weight matching as list of tuples
def max_weight_matching(B):
    matching = nx.max_weight_matching(B, maxcardinality=True, weight='weight')
    return matching

# generate and return random matching as list of tuples
def generate_random_matching(B, X, Y):
    matching = set()
    available_Y = set(Y)
    for x in X:
        if not available_Y:
            break
        matched_y = random.choice(list(available_Y))
        matching.add((x, matched_y))
        available_Y.remove(matched_y)
    return matching

# given a list of tuples representing a matching, compute total matching weight and categories with overlap
def calculate_matching_weight(B, matching, weights_dict):
    total_weight = 0
    intersects_weight = 0
    for u, v in matching:
        total_weight += B[u][v]['weight']
        if 'X' in u:
            intersects_weight += weights_dict[u][v][1]
        else:
            intersects_weight += weights_dict[v][u][1]
    return total_weight, intersects_weight

# generate num_rounds datasets/bipartite graphs, and compute 10000 random matchings for each
# compute and print average weight and categories with overlap for both max weight and random matchings  
def main():
    num_rounds = 100
    iterations_per_round = 10000
    num_students = 25
    num_mentors = 17
    meta_max_weight = 0
    meta_max_intersects = 0
    meta_random_weight = 0
    meta_random_intersects = 0
    for i in range(num_rounds):
        print('ROUND ' + str(i+1))
        students, mentors, weights_dict = generate_dataset(num_students, num_mentors)
        B, X, Y = create_bipartite_graph(students, mentors, weights_dict)
        max_weight, max_intersects = calculate_matching_weight(B, max_weight_matching(B), weights_dict)
        sum_random_matching_weights = 0
        sum_random_matching_intersects = 0
        for j in range(iterations_per_round):
            curr_weight, curr_intersects = calculate_matching_weight(B, generate_random_matching(B, X, Y), weights_dict)
            sum_random_matching_weights += curr_weight
            sum_random_matching_intersects += curr_intersects
        avg_random_matching_weight = sum_random_matching_weights / iterations_per_round
        avg_random_matching_intersects = sum_random_matching_intersects / iterations_per_round
        print('weight of max weight matching = ' + str(max_weight))
        meta_max_weight += max_weight
        print('number of intersects in max weight matching = ' + str(max_intersects))
        meta_max_intersects += max_intersects
        print('average weight of random matching = ' + str(avg_random_matching_weight))
        meta_random_weight += avg_random_matching_weight
        print('average number of intersects in random matching = ' + str(avg_random_matching_intersects))
        meta_random_intersects += avg_random_matching_intersects
    print('FINAL CUMULATIVE AVG - weight of max weight matching = ' + str(meta_max_weight / num_rounds))
    print('FINAL CUMULATIVE AVG - number of intersects in max weight matching = ' + str(meta_max_intersects / num_rounds))
    print('FINAL CUMULATIVE AVG - average weight of random matching = ' + str(meta_random_weight / num_rounds))
    print('FINAL CUMULATIVE AVG - average number of intersects in random matching = ' + str(meta_random_intersects / num_rounds))


if __name__ == "__main__":
    main()