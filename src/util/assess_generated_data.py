import sys
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from read_data import read_data

RUNS = 10  # amount of times kmeans is run to find average scores


def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))])


def get_info(fname):
    x = fname.split("-")
    d = x[0][:-1]
    c = x[1][:-1]

    return int(d), int(c)


def convert_data(data_fname, csv_fname, dataset_type):
    data_file = open(data_fname, 'w+')
    csv_file = open(csv_fname, 'r')

    d, c = get_info(dataset_type)

    data_file.write("classLast,{},{},comma\n".format(d, c))

    for line in csv_file:
        data_file.write(line)

    data_file.close()
    csv_file.close()


def assess_performance(datafile):
    all_data = read_data(datafile)
    data = all_data["data"]
    labels = all_data["labels"]

    clusters = len(set(labels))

    ari_total = 0
    ksil_total = 0

    for i in range(RUNS):
        kmeans = KMeans(n_clusters=clusters).fit(data)
        ari_total += adjusted_rand_score(labels, kmeans.labels_)
        ksil_total += silhouette_score(data, kmeans.labels_, metric="euclidean")

    sil = silhouette_score(data, labels, metric="euclidean")

    return ari_total / RUNS, sil, ksil_total / RUNS


def process_datasets(dataset_type):
    print("processing {}".format(dataset_type))
    path = os.path.join(ROOT_DIR, dataset_type)
    stats = pd.DataFrame(columns=["run", "ari", "silhouette", "kmeans-silhouette"])
    i = 1
    while os.path.isfile("%s/%s-%d.csv" % (path, dataset_type, i)):
        data_fname = "%s/%s-%d.data" % (path, dataset_type, i)

        # convert to data file if it has not happened already
        if not os.path.isfile(data_fname):
            convert_data(data_fname, "%s/%s-%d.csv" % (path, dataset_type, i), dataset_type)

        ari, sil, ksil = assess_performance(os.path.join(path, data_fname))
        stats = stats.append({'run': int(i), 'ari': ari, 'silhouette': sil, 'kmeans-silhouette': ksil}
                             , ignore_index=True)

        i += 1
    if not stats.empty:
        stats = stats.set_index(["run"])
        print(stats)
        return stats
        # stats.to_csv("{}/{}-summary.csv".format(path, dataset_type))


def main():
    all_stats = []

    for dataset_type in get_immediate_subdirectories(ROOT_DIR):
        all_stats.append((dataset_type, process_datasets(dataset_type)))

    summary = open(os.path.join(ROOT_DIR, "summary.txt"), 'w+')
    for data, stats in all_stats:
        summary.write("==={}===\n".format(data))
        summary.write(str(stats))
        summary.write("\n\n")

    summary.close()


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    main()
