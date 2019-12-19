from multi_tree import init_primitives
from main import init_toolbox
from deap import base, gp, creator
import numpy as np
import sys
import pandas as pd
import pygraphviz as pgv
import os


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def draw_individual(individual, datafile, run_type):

    g = pgv.AGraph()
    g.graph_attr['label'] = "{}:{}".format(datafile, run_type).upper()
    for i, expr in enumerate(individual):

        nodes, edges, labels = gp.graph(expr)
        node_map = dict()
        for j in range(len(nodes)):
            current = nodes[j]
            new_node = int(str(i+1)+str(current))
            nodes[j] = new_node
            node_map[current] = new_node

        for j in range(len(edges)):
            edges[j] = (node_map[edges[j][0]], node_map[edges[j][1]])

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for j in range(len(nodes)):
            n = g.get_node(node_map[j])

            # round constants to 4dp for aesthetics :-)
            try:
                labels[j] = float(labels[j])
                labels[j] = str(round(labels[j], 4))
            except:
                pass

            n.attr["label"] = labels[j]
    return g


def process_individual(datafile, run_type, row, pset):

    ind_file = "{}/{}/{}/{}_ind.txt".format(ROOT_DIR, datafile, run_type, row)
    trees = []

    for line in open(ind_file):
        if line == "\n":
            break
        string = line.strip("\n")
        trees.append(gp.PrimitiveTree.from_string(string, pset))

    ind = creator.Individual(trees)

    g = draw_individual(ind, datafile, run_type)

    g.draw("{}/{}:{}.pdf".format(OUTPUT_DIR, datafile, run_type))


def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))])


def main():
    simplest = pd.DataFrame()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    pset = gp.PrimitiveSet("MAIN", 1000, prefix="f")
    pset.context["array"] = np.array
    init_primitives(pset)

    toolbox = base.Toolbox()
    init_toolbox(toolbox, pset, 100)

    for dataset in get_immediate_subdirectories(ROOT_DIR):
        dataset_dir = os.path.join(ROOT_DIR, dataset)

        for run_type in get_immediate_subdirectories(dataset_dir):
            fname = "{}:{}".format(dataset, run_type)

            df = pd.read_csv("{}/{}.csv".format(ROOT_DIR, fname), index_col=0)
            index = df["total-nodes"].idxmin()

            if os.path.isfile("{}/{}.pdf".format(OUTPUT_DIR, fname)):
                print("{} already drawn".format(fname))
            else:

                process_individual(dataset, run_type, index, pset)

            row = df.loc[index].to_frame().transpose()
            row.insert(0, 'run', "{}".format(fname))
            row = row.drop(['base-sil', 'base-ari'], axis='columns')

            row = row.rename(columns={'best-ari': 'ari', 'best-sil': 'sil', 'best-sil-pre': 'sil-pre'})

            summary = df.describe()
            row.insert(row.columns.get_loc('ari')+1, 'mean-ari', summary.at['mean', 'best-ari'])
            row.insert(row.columns.get_loc('sil')+1, 'mean-sil', summary.at['mean', 'best-sil'])
            row.insert(row.columns.get_loc('sil-pre')+1, 'mean-sil-pre', summary.at['mean', 'best-sil-pre'])

            simplest = pd.concat([simplest, row], axis=0)

    simplest = simplest.set_index('run')
    summary_file = open("{}/simplest_summary.txt".format(OUTPUT_DIR), "w")
    summary_file.write(str(simplest))


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    if len(sys.argv) == 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        OUTPUT_DIR = ROOT_DIR

    main()
