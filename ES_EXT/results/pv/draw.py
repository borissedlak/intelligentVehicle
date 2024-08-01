import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ES_EXT import util_fgcs

row_labels_pixel = ["480p", "720p", "1080p"]
row_labels_mode = ['single', 'double']
column_labels = ["5", "10", "15", "20", "25 fps"]

for file, mini, maxi, row_labels in [('ig_CV_AGX_MAX.csv', 0.0, 1.0, row_labels_pixel), ('ig_QR_AGX_MAX.csv', 0.0, 1.0, row_labels_pixel),
                                     ('ig_LI_AGX_MAX.csv', 0.0, 1.0, row_labels_mode),
                                     ('pv_CV_AGX_MAX.csv', 0.0, 1.0, row_labels_pixel), ('pv_QR_AGX_MAX.csv', 0.0, 1.0, row_labels_pixel),
                                     ('pv_LI_AGX_MAX.csv', 0.0, 1.0, row_labels_mode),
                                     ('ig_CV_NX_MAX.csv', 0.0, 1.0, row_labels_pixel), ('ig_QR_NX_MAX.csv', 0.0, 1.0, row_labels_pixel),
                                     ('ig_LI_NX_MAX.csv', 0.0, 1.0, row_labels_mode),
                                     ('pv_CV_NX_MAX.csv', 0.0, 1.0, row_labels_pixel), ('pv_QR_NX_MAX.csv', 0.0, 1.0, row_labels_pixel),
                                     ('pv_LI_NX_MAX.csv', 0.0, 1.0, row_labels_mode)]:
    data = pd.read_csv(file, header=None)
    matrix = data.values
    interpolated = util_fgcs.interpolate_values(matrix)

    sns.set()
    plt.figure(figsize=(3.8, 3.5))  # Adjust the figure size as needed

    # Customize the heatmap, including the colormap, annot, and other parameters
    heatmap = sns.heatmap(interpolated, annot=False, fmt='.2f', cmap="crest", vmin=mini, vmax=maxi,
                          xticklabels=column_labels, yticklabels=row_labels,
                          cbar_kws={'orientation': 'vertical', 'shrink': 1.0})  # Customize colorbar label as needed

    # Set the font size for the tick labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14.0)  # Adjust fontsize as needed
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14.0)  # Adjust fontsize as needed

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14.0)  # Adjust colorbar tick label size
    # cbar.set_label('Pragmatic Value (pv)' if 'pv' in file else "Information Gain (ig)", fontsize=14.0)  # Adjust colorbar label size

    path = file.split('/')
    plt.savefig('figures/matrices/' + file.split(".")[0] + '.eps', format="eps")
    plt.show()
