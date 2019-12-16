import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

PLOT = False


def main():

    summary_file = open("%s/summary.txt" % OUTPUT_DIR, "w")
    count = 0

    for dataset in get_immediate_subdirectories(ROOT_DIR):
        summary_file.write("{1} {0} {1}\n\n".format(dataset.upper(), ("=" * 5)))
        dataset_dir = os.path.join(ROOT_DIR, dataset)
        plot_data = dict()

        for run_type in get_immediate_subdirectories(dataset_dir):
            summary_file.write(run_type.upper()+"\n")
            run_dir = os.path.join(dataset_dir, run_type)
            fname = "{}:{}".format(dataset, run_type)

            # only process run if csv does not yet exist
            if os.path.isfile("{}/{}.csvf".format(OUTPUT_DIR, fname)):
                print("{} already processed".format(fname))
                stats = pd.read_csv("{}/{}.csv".format(OUTPUT_DIR, fname), index_col=0)
            else:

                count += 1
                stats = process_run(run_dir)
                if stats is None:
                    continue
                print("processed {}, {} runs".format(fname, len(stats)))
                output_loc = "%s/%s" % (OUTPUT_DIR, fname)
                stats.to_csv(output_loc + ".csv")

            # now summarise the results
            # plot_column(stats["total-nodes"].tolist(), 10, fname)
            plot_data[run_type] = stats["total-nodes"].tolist()
            summary = stats.describe().drop("count", axis=0)
            print(summary, file=summary_file)
            print(file=summary_file)
        if PLOT:
            plot_graph(plot_data, dataset)
    print("\n{} runs processed".format(count))


def plot_graph(plots, dataset):

    fig, axes = plt.subplots(len(plots), sharex=True)
    axis = 0

    for run_type, data in plots.items():
        axes[axis].hist(data, 10, facecolor='blue', alpha=0.5)
        axes[axis].set_title("{} : {}".format(dataset, run_type))
        axis += 1
    plt.savefig("{}/{}-nodesdist.png".format(OUTPUT_DIR, dataset))


def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))])


def process_run(path):

    stats = pd.DataFrame()
    i = 1
    while os.path.isfile("%s/%d_results.txt" % (path, i)):
        result = pd.read_csv("%s/%d_results.txt" % (path, i))
        result.insert(0, 'run', i)
        stats = pd.concat([stats, result], axis=0)
        i += 1
    if stats.empty:
        return None
    stats = stats.set_index('run')
    # print("processed {} runs.".format(i))
    return stats


def plot_column(data, num_bins, title):
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]

    if len(sys.argv) == 3:
        OUTPUT_DIR = sys.argv[2]
    else:
        OUTPUT_DIR = ROOT_DIR

    main()