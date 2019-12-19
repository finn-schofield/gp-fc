import hawks
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import numpy as np
import sys


def main(ndim, nclusters, run_id):
    config = {
        "hawks": {
            "folder_name": "",
            "save_best_data": True
        },
        "dataset": {
                "num_clusters": nclusters,
                "num_dims": ndim
            },
        "ga": {
            "num_gens": 500
        },
        "constraints": {
            "overlap": {
                "threshold": 0.3,
                "limit": "lower"
            }
        }
    }

    generator = hawks.create_generator(config)
    print(generator.folder_name)
    print(generator.save_best_data)

    generator.run()
    # Get the best dataset found and it's labels
    datasets, label_sets = generator.get_best_dataset()
    # Stored as a list for multiple runs
    data, labels = datasets[0], label_sets[0]
    # Run KMeans on the data
    km = KMeans(
        n_clusters=len(np.unique(labels)), random_state=0
    ).fit(data)
    # Get the Adjusted Rand Index for KMeans on the data
    ari = adjusted_rand_score(labels, km.labels_)
    sil = silhouette_score(data, km.labels_)
    print(f"ARI: {ari}, SIL: {sil}")

    generator.plot_best_indivs(show=True)



if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

