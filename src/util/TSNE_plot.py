from read_data import read_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(fname):
    X = read_data(fname)
    X_e = TSNE().fit_transform(X['data'])

    X_a, X_b = zip(*X_e)

    plt.scatter(X_a, X_b, c=X['labels'])
    plt.title(fname.split("/ |,")[-2])
    plt.show()




