from threadpoolctl import threadpool_limits



import warnings
import sys
import re
import csv

import numpy as np
import pandas as pd
from sklearn.utils import parallel_backend

import vector_tree as vt
import multi_tree as mt

import matplotlib.pyplot as plt

from deap import gp
from deap import base
from deap import creator

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances

from ea_simple_elitism import eaSimple

from selection import *

from read_data import read_data

POP_SIZE = 1024
NGEN = 100
CXPB = 0.8
MUTPB = 0.2
ELITISM = 10
PARSIMONY = True
CMPLX = "nodes_total"  # complexity measure of individuals
BCKT = no_bucketing  # bucketing used for lexicographic parsimony pressure
BCKT_VAL = 5  # bucketing parameter
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
MT_CX = "ric"  # crossover for multi-tree {'aic', 'ric', 'sic'}
DATA_DIR = "/home/schofifinn/PycharmProjects/SSResearch/data"


def evaluate(individual, toolbox, data, k, metric, distance_vector=None, labels_true=None, plot_sil=False):
    """
    Evaluates an individuals fitness. The fitness is the clustering performance on the data
    using the specified metric.

    :param individual: the individual to be evaluated
    :param toolbox: the evolutionary toolbox
    :param data: the data to be used to evaluate the individual
    :param k: the number of clusters for k-means
    :param metric: the metric to be used to evaluate clustering performance
    :param distance_vector: a pre-computed distance vector, required for silhouette-pre metric
    :param labels_true: the ground truth cluster labels, required for ari metric
    :return: the fitness of the individual
    """

    X = REP.process_data(individual, toolbox, data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_jobs=1).fit(X)
        # optics = OPTICS().fit(X)
    labels = kmeans.labels_
    # labels = optics.labels_

    # individuals that find single cluster are unfit
    nlabels = len(set(labels))
    if nlabels == 1:
        return [-1]

    # uses precomputed distances of original data to avoid trending towards single dimension when
    # minimising silhouette.
    if metric == 'silhouette_pre':
        if distance_vector is None:
            raise ValueError("Must provide distance vector for silhouette-pre metric.")
        if plot_sil:
            silhouette = silhouette_samples(distance_vector, labels, metric='precomputed')
            plot_silhouette(silhouette, 'silhouette-pre')
        return [silhouette_score(distance_vector, labels, metric='precomputed')]
    elif metric == 'silhouette':
        if plot_sil:
            silhouette = silhouette_samples(X, labels)
            plot_silhouette(silhouette, 'silhouette')
        return [silhouette_score(X, labels, metric='euclidean')]
    elif metric == 'ari':
        if labels_true is None:
            raise ValueError("Must provide ground truth labels for ARI")
        return [adjusted_rand_score(labels_true, labels)]
    elif metric == 'intra':
        raise NotImplementedError("intra metric not implemented")
        # return [-kmeans.inertia_]
    else:
        raise Exception("invalid metric: {}".format(metric))


def eval_complexity(individual, measure):
    """
    Evaluates the complexity of an individual using the given measure.

    :param individual: the individual to be evaluated
    :param measure: the measure of individual complexity
    :return: individuals complexity
    """

    if REP is mt:
        con_fts = [str(tree) for tree in individual]
    elif REP is vt:
        con_fts = vt.parse_tree(individual)
    else:
        raise Exception("Invalid representation")

    if measure == "cf_count":
        complexity = len(con_fts)
    elif measure == "unique_fts":
        unique_fts = set()
        pat = re.compile("f[\\d]+")

        for cf in con_fts:
            unique_fts.update(re.findall(pat, cf))

        complexity = len(unique_fts)
    elif measure == "nodes_avg" or measure == "nodes_total":
        total_nodes = 0
        for cf in con_fts:
            total_nodes += 1  # root node
            for i in range(len(cf)):
                # each node is preceded by either a comma or an opening bracket (except root)
                if cf[i] == ',' or cf[i] == '(':
                    total_nodes += 1
        if measure == "nodes_avg":
            complexity = total_nodes / len(con_fts)
        else:
            complexity = total_nodes
    else:
        raise Exception("Invalid complexity metric: %s" % measure)

    return complexity


def plot_silhouette(silhouette, title):
    plt.hist(silhouette, bins=30)
    plt.title(title)
    plt.show()


def write_ind_to_file(ind, run_num, results):
    """
    Writes the attributes of an individual to a csv file.

    :param run_num: the number of the current run
    :param ind: the individual
    :param results: a dictionary of results, titles to values
    """

    line_list = []

    # add constructed features to lines
    if REP is mt:
        for cf in [str(tree) for tree in ind]:
            line_list.append(cf + "\n")
    elif REP is vt:
        for cf in vt.parse_tree(ind):
            line_list.append(cf + "\n")
    else:
        raise Exception("Invalid representation")

    line_list.append("\n")

    fl = open("%d_ind.txt" % run_num, 'w')
    fl.writelines(line_list)
    fl.close()

    csv_columns = results.keys()
    csv_file = "%d_results.txt" % run_num

    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(results)


def init_toolbox(toolbox, pset, n_trees):
    """
    Initialises the toolbox with evolutionary operators.

    :param toolbox: the toolbox to initialise
    :param pset: primitive set for evolution
    """
    if REP is mt:
        mt.init_toolbox(toolbox, pset, MT_CX, n_trees)
    if REP is vt:
        mt.init_toolbox(toolbox, pset, MT_CX)

    if PARSIMONY:
        toolbox.register("eval_complexity", eval_complexity, measure=CMPLX)
        toolbox.register("bucket", BCKT, BCKT_VAL)
        toolbox.register("select", parsimony_tournament, tournsize=7, toolbox=toolbox)
    else:
        toolbox.register("select", tools.selTournament, tournsize=7)


def init_stats():
    """
    Initialises a MultiStatistics object to capture data.

    :return: the MultiStatistics object
    """
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    if CMPLX == 'unique_fts':
        unique_stats = tools.Statistics(lambda ind: eval_complexity(ind, "nodes_total"))
        stats = tools.MultiStatistics(fitness=fitness_stats, unique_fts=unique_stats)
    else:
        nodes_stats = tools.Statistics(lambda ind: eval_complexity(ind, "nodes_total"))
        stats = tools.MultiStatistics(fitness=fitness_stats, total_nodes=nodes_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return stats


def final_evaluation(best, data, labels, num_classes, toolbox, print_output=True):
    """
    Performs a final performance evaluation on an individual.

    :param best: the individual to evaluate
    :param data: the dataset associated with the individual
    :param labels: the ground-truth labels of the dataset
    :param num_classes: the number of classes of the dataset
    :param toolbox: the evolutionary toolbox
    :param print_output: whether or not to print output
    :return: a dictionary of results, titles to values
    """
    kmeans = KMeans(n_clusters=num_classes, random_state=SEED).fit(data)
    labels_pred = kmeans.labels_
    baseline_ari = adjusted_rand_score(labels, labels_pred)
    baseline_silhouette = silhouette_score(data, labels_pred, metric="euclidean")
    silhouette = silhouette_samples(data, labels_pred)
    # plot_silhouette(silhouette, 'baseline silhouette')

    best_ari = evaluate(best, toolbox, data, num_classes, "ari", labels_true=labels)[0]
    best_silhouette = evaluate(best, toolbox, data, num_classes, "silhouette")[0]
    cfs = eval_complexity(best, "cf_count")
    unique = eval_complexity(best, "unique_fts")
    nodes = eval_complexity(best, "nodes_total")

    if print_output:
        print("\nConstructed features: %d" % cfs)
        print("Unique features: %d\n" % unique)
        print("Best ARI: %f \nBaseline ARI: %f\n" % (best_ari, baseline_ari))
        print("Best silhouette: %f \nBaseline silhouette: %f" % (best_silhouette, baseline_silhouette))

    return {"constructed-fts": cfs, "unique-fts": unique, "total-nodes": nodes, "best-ari": best_ari,
            "base-ari": baseline_ari, "best-sil": best_silhouette,  "best-sil-pre": best.fitness.values[0],
            "base-sil": baseline_silhouette}


def plot_stats(logbook):
    """
    Generates plots of the statistics gathered from the evolutionary process.
    :param logbook: a logbook of the statistics.
    """
    gen = logbook.select("gen")
    fit_max = logbook.chapters["fitness"].select("max")
    nodes_avg = logbook.chapters["total_nodes"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, nodes_avg, "r-", label="Average Nodes")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def main(datafile, run_num):

    random.seed(SEED)
    all_data = read_data("%s/%s.data" % (DATA_DIR, datafile))
    data = all_data["data"]
    labels = all_data["labels"]

    num_classes = len(set(labels))
    print("%d classes found." % num_classes)
    distance_vector = pairwise_distances(data)

    num_instances = data.shape[0]
    num_features = data.shape[1]

    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # set up toolbox
    toolbox = base.Toolbox()
    n_trees = min(num_classes // 2, num_features // 2)
    init_toolbox(toolbox, pset, n_trees)

    toolbox.register("evaluate", evaluate, toolbox=toolbox, data=data, k=num_classes,
                     metric='silhouette_pre', distance_vector=distance_vector)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = init_stats()

    pop, logbook = eaSimple(pop, toolbox, CXPB, MUTPB, ELITISM, NGEN, stats, halloffame=hof, verbose=True)

    for chapter in logbook.chapters:
        logbook_df = pd.DataFrame(logbook.chapters[chapter])
        logbook_df.to_csv("%s_%d.csv" % (chapter, run_num), index=False)

    best = hof[0]
    res = final_evaluation(best, data, labels, num_classes, toolbox)
    # evaluate(best, toolbox, data, num_classes, 'silhouette_pre', distance_vector=distance_vector,
    #          plot_sil=True)
    write_ind_to_file(best, run_num, res)

    return pop, stats, hof


"""
[seed] [data file] [{parsimony, noparsimony}]
"""
if __name__ == "__main__":
    SEED = int(sys.argv[1])
    run_type = sys.argv[3]

    PARSIMONY = False if run_type.startswith('no') else True

    if 'unique' in run_type:
        CMPLX = 'unique_fts'
    else:
        CMPLX = 'nodes_total'

    if run_type.endswith('aic'):
        MT_CX = 'aic'
    elif run_type.endswith('sic'):
        MT_CX = 'sic'
    else:
        MT_CX = 'ric'
    with threadpool_limits(limits=1, user_api='blas'):
        main(sys.argv[2], SEED)



