import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ES_EXT import util_fgcs

row_labels = [120, 180, 240, 300, 360, 420]
column_labels = [14, 18, 22, 26, 30]

for file, mini, maxi in [('ig_CV_NX_MAX.csv', 0.0, 1.0), ('ig_QR_NX_MAX.csv', 0.0, 1.0), ('ig_LI_NX_MAX.csv', 0.0, 1.0)]:
    data = pd.read_csv(file, header=None)
    matrix = data.values
    interpolated = util_fgcs.interpolate_values(matrix)

    sns.set()
    plt.figure(figsize=(3.8, 3.5))  # Adjust the figure size as needed

    # Customize the heatmap, including the colormap, annot, and other parameters
    heatmap = sns.heatmap(interpolated, annot=False, fmt='.2f', cmap="crest", vmin=mini, vmax=maxi)
    # xticklabels=column_labels, yticklabels=row_labels)

    path = file.split('/')
    plt.savefig('figures/' + file.split(".")[0] + '.eps', format="eps")
    plt.show()
