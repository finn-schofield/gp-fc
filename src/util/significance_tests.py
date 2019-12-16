import sys
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import os

measures = ['unique-fts', 'total-nodes', 'best-ari', 'best-sil', 'best-sil-pre']


def main():

    datafiles = dict()
    mn_rows = []
    wc_rows = []

    for file in sorted(os.listdir(ROOT_DIR)):
        if file.endswith(".csv"):
            datafile = file.split(":")[0]
            if datafile in datafiles:
                continue

            try:
                null_df = pd.read_csv("{}/{}:{}.csv".format(ROOT_DIR, datafile, NULL_FILE), index_col=0)
                alt_df = pd.read_csv("{}/{}:{}.csv".format(ROOT_DIR, datafile, ALT_FILE), index_col=0)
            except FileNotFoundError:
                continue

            datafiles[datafile] = (null_df, alt_df)

    for dataname, dataframes in datafiles.items():

        wc_row = [dataname]
        mn_row = [dataname]

        for measure in measures:
            null_data = dataframes[0][measure].to_list()
            alt_data = dataframes[1][measure].to_list()

            try:
                _, wc_p = wilcoxon(alt_data, null_data)
                _, mn_p = mannwhitneyu(null_data, alt_data)
            except ValueError as e:
                if "length" in str(e):
                    raise ValueError("{} lengths not equal".format(dataname)) from e
                else:
                    wc_p = 1.0
                    mn_p = 1.0

            wc_row.append(wc_p)
            mn_row.append(mn_p)

        wc_rows.append(wc_row)
        mn_rows.append(mn_row)

    measures.insert(0, 'datafile')

    wc = pd.DataFrame(wc_rows, columns=measures)
    wc = wc.set_index('datafile')

    mn = pd.DataFrame(mn_rows, columns=measures)
    mn = mn.set_index('datafile')

    fl = open("{}/{}-{}-significance.txt".format(ROOT_DIR, NULL_FILE, ALT_FILE), 'w')
    fl.write("Wilcoxon Results:\n\n")
    fl.write(str(wc))

    fl.write("\n\nMann-Whitney Results:\n\n")
    fl.write(str(mn))


if __name__ == "__main__":
    ROOT_DIR = sys.argv[1]
    NULL_FILE = sys.argv[2]
    ALT_FILE = sys.argv[3]
    main()
